import streamlit as st
import requests

# Run Using
# streamlit run ui_fe.py

st.set_page_config(page_title="AgriGuru Chatbot", layout="centered")

# Mapping endpoint names to URLs
ENDPOINTS = {
    "Crop Recommendation": "http://localhost:8000/ask",  # Replace with actual endpoint
    "Crop Detection": "http://localhost:8000/detect-disease",
    "Disease Analysis": "http://localhost:8000/detect-disease",
    "Crop Price": "http://localhost:8000/ask"
}

# Sidebar to choose page
page = st.sidebar.radio("Choose Chatbot Page", list(ENDPOINTS.keys()))

# Chat History State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        key: [] for key in ENDPOINTS.keys()
    }

st.title(page)

# Chat UI
for sender, msg in st.session_state.chat_history[page]:
    if sender == "user":
        st.markdown(f"<div style='text-align: left;color: black; background: #e0f7fa; padding: 8px;border:1px solid #b0a7ba; border-radius: 5px; margin-bottom: 5px;'>🧑‍💻 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;color: black; background: #fce4ec; padding: 8px;border:1px solid #fc64ec; border-radius: 5px; margin-bottom: 5px;margin-left: 200px'>🤖 {msg}</div>", unsafe_allow_html=True)

# Input Section
if page == "Disease Analysis":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("Send"):
        if uploaded_file:
            st.session_state.chat_history[page].append(("user", uploaded_file.name))
            with st.spinner("Sending image..."):
                try:
                    response = requests.post(
                        ENDPOINTS[page],
                        timeout=300,
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())}
                    )
                    reply = response.json()['diagnosis'][0]
                    reply.pop("path")
                    if reply['status']!="Healthy":
                        reply['disease']=reply['label']
                    reply.pop('label')
                except Exception as e:
                    reply = f"❌ Error: {str(e)}"
            st.session_state.chat_history[page].append(("bot", reply))
            st.rerun() # Optional here, only if needed to rerun input
        else:
            st.warning("Please upload an image")
elif page == "Crop Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("Send"):
        if uploaded_file:
            st.session_state.chat_history[page].append(("user", uploaded_file.name))
            with st.spinner("Sending image..."):
                try:
                    response = requests.post(
                        ENDPOINTS[page],
                        timeout=300,
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())}
                    )
                    reply = response.json()['diagnosis'][0]['crop']
                except Exception as e:
                    reply = f"❌ Error: {str(e)}"
            st.session_state.chat_history[page].append(("bot", reply))
            st.rerun() # Optional here, only if needed to rerun input
        else:
            st.warning("Please upload an image")

else:
    user_input = st.text_input("Enter your message", key=page)
    if st.button("Send", key=f"send_{page}"):
        if user_input.strip():
            st.session_state.chat_history[page].append(("user", user_input))
            with st.spinner("Sending message..."):
                try:
                    response = requests.post(
                        ENDPOINTS[page],
                        timeout=300,
                        # json={"message": user_input}
                        json={"question": user_input}
                    )
                    reply = response.json()['answer']['result']
                except Exception as e:
                    reply = f"❌ Error: {str(e)}"
            st.session_state.chat_history[page].append(("bot", reply))
            st.rerun()
  # Optional – helps re-render but not always necessary

