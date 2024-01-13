import streamlit as st
from home import face_rec
import av
from streamlit_webrtc import webrtc_streamer
import numpy as np

#st.set_page_config(page_title='Registration Form')

st.subheader('Registration Form')

#init registration form
registration_form=face_rec.RegistrationForm()

# Step 1: Collect person name and role
# form
person_name=st.text_input(label='Name',placeholder='First & Last Name')
role= st.selectbox(label='Select your Role',options=('Student','Teacher'))




# Step 2: Collect facial embedding of that person

def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    # two step process
    # 1st step save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:             
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,webrtc_streamer(
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
))


# Step 3: Save data in redis database.
if st.button('Submit'):
    return_value=registration_form.save_data_in_redis_db(person_name,role)
    if return_value == True:
        st.success(f"{person_name} registered successfully")

    elif return_value == 'name_false':
        st.error('name_is_not_correct!')
    
    elif return_value == 'file_false':
        st.error('Face_embedding_not_present. Please refresh and execute again.')