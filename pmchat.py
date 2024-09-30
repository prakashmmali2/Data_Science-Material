from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(que):
  response = chat.send_message(que , stream=True)
  return response

st.set_page_config(page_title="Q&A PmChat")
st.header("Let's chat with PMchatBot")

if "chat_history" not in st.session_state:
  st.session_state["chat_history"] = []

input = st.text_input("You : " , key = "input")
Submit = st.button("Ask the Question : ")

if Submit and input:
  response = get_gemini_response(input)
  st.session_state["chat_history"].append(("You" , input))
  st.subheader("The Response is : ")
  for chunk in response :
    st.write(chunk.text)
    st.session_state["chat_history"].append(("Bot",chunk.text))

st.subheader("The Chat History")

for role,text in st.session_state["chat_history"]:
  st.write(f"{role}:{text}")