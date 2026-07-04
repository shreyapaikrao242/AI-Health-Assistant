import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Health Assistant")

st.write(
    "Ask me questions about heart disease, diabetes, healthy lifestyle and general health."
)

question = st.text_area(
    "Ask your health question"
)

if st.button("Ask AI"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            chat = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            
                            """
You are AI Health Assistant developed by Shreya Paikrao.

Your role is to educate users about heart disease, diabetes,
healthy lifestyle, nutrition, exercise and general wellness.

Rules:

1. Only answer health-related questions.

2. If a user asks something unrelated
(example: movies, programming, politics, jokes),
politely reply:

'I am an AI Health Assistant designed to answer only health-related questions.'

3. Never diagnose diseases.

4. Never prescribe medicines.

5. Encourage users to consult qualified healthcare professionals.

6. Keep explanations simple so that beginners can understand them.

7. Be friendly and professional.
"""
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )

        answer = chat.choices[0].message.content

        st.success(answer)