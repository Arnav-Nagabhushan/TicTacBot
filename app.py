import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from teachable_machine import TeachableMachine
from xo import ai
from xo.board import Board

st.set_page_config(page_title="TicTacBot", page_icon="🤖", layout="centered")
st.title("TicTacBot 🎮")
st.caption(":rainbow[AI-Powered] Tic Tac Toe")

# Collapsible instructions expander for beginners
with st.expander("📖 How to Play TicTacBot"):
    st.write("1. Make your move (**X**) on your physical board.")
    st.write("2. Take a photo using your camera or upload an image.")
    st.write("3. Watch :rainbow[TicTacBrain] scan the board and calculate the best response.")
    st.write("4. Copy the AI's move (**O**) to your board and click Next Turn!")
    st.write("5. Don't be upset if you cannot win, the difficulty set is 'unbeatable'! 😈")

current_dir = os.path.dirname(os.path.abspath(__file__)) # Define current directory
yolo_path = os.path.join(current_dir, "best.pt")
tm_path = os.path.join(current_dir, "keras_model.h5")
labels_path = os.path.join(current_dir, "labels.txt")

@st.cache_resource # Load heavy AI Models, then cache (remember)
def load_all_models():
    yolo = YOLO(yolo_path)
    tm = TeachableMachine(model_path=tm_path, labels_file_path=labels_path)
    return yolo, tm

try:
    yolo_model, tm_model = load_all_models()
except Exception as e:
    st.error(f"TicTacBot has encountered a critical error: Model files not found. Details: {e}")
    st.stop()  # Stops the app completely so it never crashes with a NameError later!

if "turn" not in st.session_state: # Turn = 1, if not defined previously
    st.session_state.turn = 1

def sort_by_y(box): # Default Python function to sort from top to bottom
    return box[1] 

st.divider()
st.subheader(f"Turn {st.session_state.turn}: Capture Board")

tab_camera, tab_upload = st.tabs(["📷 Camera", "📁 Upload Image"])
image_to_process = None

with tab_camera:
    camera_photo = st.camera_input("Take a photo of your board", key=f"camera_{st.session_state.turn}")
    if camera_photo is not None:
        image_to_process = camera_photo

with tab_upload:
    uploaded_file = st.file_uploader("Upload an image of your board", type=["jpg", "png", "jpeg"], key=f"upload_{st.session_state.turn}")
    if uploaded_file is not None:
        image_to_process = uploaded_file

if image_to_process is not None:
    try:
        with st.spinner("Analyzing board with AI..."):
            image = Image.open(image_to_process).convert("RGB")
            
            results = yolo_model(image, conf=0.1)
            boxes = results[0].boxes.xyxy.cpu().numpy() # Extract cell coordinates
            
            if len(boxes) != 9:
                st.warning(f"Detected {len(boxes)} cells instead of 9! Try again using a different angle/lighting.")
            else:
                st.toast("Board scanned successfully!", icon="🎯")
                boxes = sorted(boxes, key=sort_by_y)
                
                board = [
                    [".", ".", "."],
                    [".", ".", "."],
                    [".", ".", "."]
                ]
                
                cell_index = 0
                
                # Loop through each detected cell box
                for box in boxes:
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    
                    cell_image = image.crop((x1, y1, x2, y2)) # Crop particular cell
                    cell_image.save("temporary_cell.jpg")
                    
                    result = tm_model.classify_image("temporary_cell.jpg")
                    class_index = result["class_index"]

                    # Convert number (0-8) to (row, col)
                    row = cell_index // 3
                    col = cell_index % 3
                    
                    if class_index == 0:
                        board[row][col] = "X"
                    elif class_index == 1:
                        board[row][col] = "O"
                    else:
                        board[row][col] = "." # Empty
                        
                    cell_index = cell_index + 1
                    
                st.subheader("Detected Board (Your Move as X):")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if board[0][0] == "X": st.info("❌ X")
                    elif board[0][0] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c2:
                    if board[0][1] == "X": st.info("❌ X")
                    elif board[0][1] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c3:
                    if board[0][2] == "X": st.info("❌ X")
                    elif board[0][2] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")

                c4, c5, c6 = st.columns(3)
                with c4:
                    if board[1][0] == "X": st.info("❌ X")
                    elif board[1][0] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c5:
                    if board[1][1] == "X": st.info("❌ X")
                    elif board[1][1] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c6:
                    if board[1][2] == "X": st.info("❌ X")
                    elif board[1][2] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")

                c7, c8, c9 = st.columns(3)
                with c7:
                    if board[2][0] == "X": st.info("❌ X")
                    elif board[2][0] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c8:
                    if board[2][1] == "X": st.info("❌ X")
                    elif board[2][1] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")
                with c9:
                    if board[2][2] == "X": st.info("❌ X")
                    elif board[2][2] == "O": st.warning("⭕ O")
                    else: st.write("⚫ [Empty]")

                # Convert to 2-D array string (xo requires this format)    
                board_string = board[0][0] + board[0][1] + board[0][2] + \
                               board[1][0] + board[1][1] + board[1][2] + \
                               board[2][0] + board[2][1] + board[2][2]
                               
                # Check if the board is full
                if "." not in board_string:
                    st.balloons()
                    st.info("Game Over! The board is full.")
                else:
                    # Ask the 'xo' AI library for O's best move
                    game_board = Board.fromstring(board_string.lower())
                    ai_result = ai.evaluate(game_board, "o")
                    
                    move = ai_result.positions[0]
                    ai_row = move[0]
                    ai_col = move[1]
                    
                    board[ai_row - 1][ai_col - 1] = "O" # -1 because Python numbering starts from 0
                    
                    st.subheader(f"🤖 AI Played O at Row {ai_row}, Column {ai_col}")
                    st.success("Updated Board (Copy this AI move to your physical board!):")
                    
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        if board[0][0] == "X": st.info("❌ X")
                        elif board[0][0] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d2:
                        if board[0][1] == "X": st.info("❌ X")
                        elif board[0][1] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d3:
                        if board[0][2] == "X": st.info("❌ X")
                        elif board[0][2] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")

                    d4, d5, d6 = st.columns(3)
                    with d4:
                        if board[1][0] == "X": st.info("❌ X")
                        elif board[1][0] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d5:
                        if board[1][1] == "X": st.info("❌ X")
                        elif board[1][1] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d6:
                        if board[1][2] == "X": st.info("❌ X")
                        elif board[1][2] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")

                    d7, d8, d9 = st.columns(3)
                    with d7:
                        if board[2][0] == "X": st.info("❌ X")
                        elif board[2][0] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d8:
                        if board[2][1] == "X": st.info("❌ X")
                        elif board[2][1] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                    with d9:
                        if board[2][2] == "X": st.info("❌ X")
                        elif board[2][2] == "O": st.warning("⭕ O")
                        else: st.write("⚫ [Empty]")
                        
                    st.toast("AI move calculated successfully!", icon="🤖")
                    st.balloons()
                    
                    if st.button("Next Turn ➔"):
                        st.session_state.turn += 1
                        st.rerun()
    except Exception as e:
        st.error(f"TicTacBot has encountered a critical error: {e}")
        st.stop()

st.divider()
if st.button("Reset Game"):
    st.session_state.turn = 1
    st.rerun()
