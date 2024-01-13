
import streamlit as st
from home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

#st.set_page_config(page_title='Predictions')

st.subheader('Real-Time Attendance System')

#Retrieve darta
with st.spinner('Retrieving data from Redis... '):
    redis_face_db=face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)

st.spinner('Data Retrieved successfully')

#time
waittime=30#time in seconds
settime=time.time()
realtimepred=face_rec.RealTimePred()



# Real Time Prediction
#streamlit webrtc

def video_frame_callback(frame):
    global settime
    img = frame.to_ndarray(format="bgr24")
    #operations that you can perform on the array

    pred_img=realtimepred.face_predictions(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)

    timenow=time.time()
    difftime=timenow-settime
    if difftime >= waittime:
        realtimepred.savelogs_redis()       # save data to redis every 30 secs
        settime=time.time()                 # reset time
        print('Save data to redis database...')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="RealTimePrediction", video_frame_callback=video_frame_callback,
 webrtc_streamer(
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
))



