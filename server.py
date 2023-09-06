import asyncio
import websockets
import subprocess
import os
import logging
import time
import pandas as pd
import pickle
from collections import defaultdict
from speaker_recognition_classification import extract_features_and_predict
from gesturer_recognition_classification import extract_windows_and_predict




async def handle_message(websocket, path):
    # logging.basicConfig(level=logging.DEBUG)
    
    while True:
        message = await websocket.recv()
        if message == '1000' or message == bytearray(1000):
            print('Close frame received')
            continue

        print("Message Received:",message)
        print("Message Length:",len(message))
        


        if message[0] == 1: 

            audio_filepath = os.path.join(os.getcwd(),f'audio_raw_{str(time.time())[:10]+str(time.time())[12:]}.3gp')
            with open(audio_filepath, "wb") as binary_file: binary_file.write(message[1:])   # Remove header byte and write audio file bytes to file
            await websocket.send("RECEIVED AUDIO FILE")
            audio_confidence_vals = extract_features_and_predict(convertRawAudio(audio_filepath))
            await websocket.send(f"AUDIO PREDICTED PROBABILITIES {str(audio_confidence_vals)}")
            with open('audio_confidence_vals.pickle', 'wb') as f: pickle.dump(audio_confidence_vals, f)
            print("Audio Confidence Values:", audio_confidence_vals)
            try:
                with open('gesture_confidence_vals.pickle', 'rb') as f: gesture_confidence_vals = pickle.load(f)
                predicted_user = 0
                confidence = 0
                for user_id in range(1,5+1):
                    user_confidence = gesture_confidence_vals[user_id]*audio_confidence_vals[user_id]
                    if user_confidence >= confidence:
                        predicted_user = user_id
                        confidence = user_confidence
                        print(user_id,user_id,user_confidence)
                        
                if predicted_user != 0:
                    await websocket.send(f"Predicted User ID: {predicted_user}")
                    print(f"Predicted User ID: {predicted_user}")
                    os.remove('audio_confidence_vals.pickle')
                    os.remove('gesture_confidence_vals.pickle')
            except:
                print("Missing gesture data.")
                await websocket.send(f"Missing gesture data.")
            
        else: 
            await websocket.send("RECEIVED ACCEL FILE")
            
            # print(message[1:].decode("utf-8").split('$'))     #['1,4342134,0.333,0.53535,0.5353535','0,4342134,0.333,0.53535,0.5353535'...]
            gesture_data = [s.split(',') for s in message[1:].decode("utf-8").split('$')]    # [['1',4342134','0.333','0.53535','0.5353535'],['0','4342134','0.333'',0.53535','0.5353535']...]
            time_stamped_data = defaultdict(dict)
            try:
                for data_type, timestamp, x, y, z in gesture_data:
                    if data_type == '0':
                        time_stamped_data[timestamp]['accel_x'] = x
                        time_stamped_data[timestamp]['accel_y'] = y
                        time_stamped_data[timestamp]['accel_z'] = z
                    if data_type == '1':
                        time_stamped_data[timestamp]['gyro_x'] = x
                        time_stamped_data[timestamp]['gyro_y'] = y
                        time_stamped_data[timestamp]['gyro_z'] = z
            except: print('Not enough values to unpack - end of bytestream.')
                    
            data = pd.DataFrame.from_dict(time_stamped_data, orient='index')  
            file_name = f"gesture_data_{str(time.time())[:10]+str(time.time())[12:]}.csv"
            data.to_csv(file_name)
            gesture_confidence_vals = extract_windows_and_predict(pd.read_csv(file_name))
            await websocket.send(f"GESTURE PREDICTED PROBABILITIES {str(gesture_confidence_vals)}")
            with open('gesture_confidence_vals.pickle', 'wb') as f: pickle.dump(gesture_confidence_vals, f)
            print("Gesture Confidence Values:", gesture_confidence_vals)
            try:
                with open('audio_confidence_vals.pickle', 'rb') as f: audio_confidence_vals = pickle.load(f)
                predicted_user = 0
                confidence = 0
                for user_id in range(1,5+1):
                    user_confidence = gesture_confidence_vals[user_id]*audio_confidence_vals[user_id]
                    if user_confidence >= confidence:
                        predicted_user = user_id
                        confidence = user_confidence
                        print(user_id,user_id,user_confidence)
                        
                if predicted_user != 0:
                    await websocket.send(f"Predicted User ID: {predicted_user}")
                    print(f"Predicted User ID: {predicted_user}")
                    os.remove('audio_confidence_vals.pickle')
                    os.remove('gesture_confidence_vals.pickle')

            except:
                print("Missing audio data.")
                await websocket.send(f"Missing audio data.")

# Takes file path of input audio file and outputs file path of output audio file
def convertRawAudio(filePath):
    outputPath =f'audio_wav_{str(time.time())[:10]+str(time.time())[12:]}.wav'
    cmd = ['ffmpeg', '-i', filePath, outputPath]
    subprocess.run(cmd)
    print(f"Converted raw audio file to '{outputPath}'")
    return outputPath

async def main():
    # Start the WebSocket server on port 8765

    print('Websocket Started')
    async with websockets.serve(handle_message, "192.168.1.9", 8765):
        await asyncio.Future()  # Run forever

asyncio.run(main())

