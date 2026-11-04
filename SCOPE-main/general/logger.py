
from datetime import datetime
import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
file_path = os.path.join(log_dir, f"{timestamp}.log")


LOG_AREA = st.empty()

def log_print(*args, sep=' ', end='\n'):
    """
    print, write out to log file, and display in Streamlit app.
    
    """
    output = sep.join(str(arg) for arg in args) + end
    print(output,end='')

    with open(file_path, "a") as f:
        f.write(output)

   
    # if "log_buffer" not in st.session_state:
    #     st.session_state.log_buffer = []
    # st.session_state.log_buffer.append(output)
    # # log_area.text(''.join(st.session_state.log_buffer))
    # LOG_AREA.code(  
    #     ''.join(st.session_state.log_buffer),
    #     language="bash"
    # )