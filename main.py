"""
    Main APP and UI
"""
# pylint: disable=C0301,C0103,C0303,C0411,C0304

import streamlit as st

from utils_streamlit import streamlit_hack_remove_top_space, hide_footer, create_sticky_header
from utils.app_logger import init_streamlit_logger
from ui.dialog_storage import DialogStorage, DialogRole
from backend.backend import Backend

init_streamlit_logger()

# ------------------------------- Session

SESSION_BACKEND = 'backend_session'
if SESSION_BACKEND not in st.session_state:
    st.session_state[SESSION_BACKEND] = Backend()
backend = st.session_state[SESSION_BACKEND]

SESSION_DIALOG_STORAGE = 'dialog_storage'
if SESSION_DIALOG_STORAGE not in st.session_state:
    st.session_state[SESSION_DIALOG_STORAGE] = DialogStorage()
dialog_storage = st.session_state[SESSION_DIALOG_STORAGE]

SESSION_TOKENS_COUNT = 'tokens_count'
if SESSION_TOKENS_COUNT not in st.session_state:
    st.session_state[SESSION_TOKENS_COUNT] = 0

SESSION_QUERY_EXAMPLES = 'query_examples'
if SESSION_QUERY_EXAMPLES not in st.session_state:
    st.session_state[SESSION_QUERY_EXAMPLES] = []

# ------------------------------- UI
st.set_page_config(page_title= "Demo POC", layout="wide")

streamlit_hack_remove_top_space()
hide_footer()

header = st.container()
create_sticky_header(header, "Demo")

progress = st.empty()

with st.sidebar:
    with st.form(key="uploadContent", clear_on_submit=True):
        uploaded_index_file = st.file_uploader(
            'Upload chat index',
            type=["zip"],
            accept_multiple_files= False,
            key="uploaded_index_zip"
        )
        load_button = st.form_submit_button(label="Upload")
    query_examples = st.expander(label="Query examples", expanded= True)
    tokens_used_container = st.empty()

def refresh_tokens_used():
    """Refresh tokens used"""
    tokens_used_container.markdown(f'Tokens used {st.session_state[SESSION_TOKENS_COUNT]}')

def exec_system_prompt(command : str):
    """Run system prompt"""
    if command == "/clear":
        dialog_storage.clear()

refresh_tokens_used()

if not st.session_state[SESSION_QUERY_EXAMPLES]:
    st.session_state[SESSION_QUERY_EXAMPLES] = backend.get_query_examples()
query_examples.markdown('\n'.join(st.session_state[SESSION_QUERY_EXAMPLES]))

if load_button:
    if not uploaded_index_file:
        progress.markdown('Please add index file to upload')
        st.stop()
    backend.upload_index(uploaded_index_file)

prompt = st.chat_input("Your question or command")
if prompt:
    if str(prompt).startswith("/"):
        exec_system_prompt(prompt)
        st.stop()
    dialog_storage.add_message(DialogRole.USER, prompt)
    #answer = backend.get_answer(prompt)

    search_result = backend.get_chunks(prompt)
    if search_result:
        for s in search_result:
            dialog_storage.add_message(DialogRole.ASSISTANT, s.content)
    else:
        dialog_storage.add_message(DialogRole.ASSISTANT, "Sorry, I have no answer to your question")


    #dialog_storage.add_message(DialogRole.ASSISTANT, answer.answer)
    #st.session_state[SESSION_TOKENS_COUNT] += answer.tokens_used
    refresh_tokens_used()

dialog_items = dialog_storage.get_message_list()
for dialog_item in dialog_items:
    with st.chat_message(dialog_item.role):
        st.write(dialog_item.msg)
