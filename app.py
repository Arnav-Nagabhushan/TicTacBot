import os

# Keep TensorFlow and PyTorch from starting too many threads on the cloud.
# Put these settings before importing YOLO or TensorFlow.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import faulthandler
faulthandler.enable()

import streamlit as st
from PIL import Image

st.set_page_config(page_title="TicTacBot", page_icon="🤖", layout="centered")
st.title("TicTacBot 🎮")
st.caption("AI-Powered Tic Tac Toe")

with st.expander("📖 How to Play TicTacBot", expanded=True):
    st.write("1. Make your move as X on your physical board.")
    st.write("2. Take a clear photo of the whole board or upload an image.")
    st.write("3. TicTacBot detects the cells and identifies X, O, and empty spaces.")
    st.write("4. Copy the AI's O move onto your board, then click Next Turn.")

# Find the model files in the same folder as app.py.
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, "best.pt")
tm_path = os.path.join(current_dir, "keras_model.h5")
labels_path = os.path.join(current_dir, "labels.txt")
temporary_cell_path = os.path.join(current_dir, "temporary_cell.jpg")


# Load the models only when an image is analysed, so the page can start first.
@st.cache_resource
def load_all_models():
    print("DEBUG 1: Importing YOLO", flush=True)
    from ultralytics import YOLO

    print("DEBUG 2: Importing Teachable Machine", flush=True)
    from teachable_machine import TeachableMachine

    print("DEBUG 3: Loading YOLO model", flush=True)
    yolo = YOLO(yolo_path)
    print("DEBUG 4: YOLO model loaded", flush=True)

    print("DEBUG 5: Loading Teachable Machine model", flush=True)
    tm = TeachableMachine(model_path=tm_path, labels_file_path=labels_path)
    print("DEBUG 6: Teachable Machine model loaded", flush=True)

    return yolo, tm


# Find the centre of a cell from top to bottom.
def sort_by_y(box):
    return (box[1] + box[3]) / 2


# Find the centre of a cell from left to right.
def sort_by_x(box):
    return (box[0] + box[2]) / 2


if "turn" not in st.session_state:
    st.session_state.turn = 1

if "game_id" not in st.session_state:
    st.session_state.game_id = 1

st.divider()
st.subheader("Turn " + str(st.session_state.turn) + ": Capture Your Board")

# Let the user take a photo or upload one.
camera_tab, upload_tab = st.tabs(["📷 Camera", "📁 Upload Image"])
image_to_process = None

with camera_tab:
    camera_photo = st.camera_input(
        "Take a photo of the whole board",
        key="camera_" + str(st.session_state.game_id) + "_" + str(st.session_state.turn)
    )
    if camera_photo is not None:
        image_to_process = camera_photo

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a board image",
        type=["jpg", "jpeg", "png"],
        key="upload_" + str(st.session_state.game_id) + "_" + str(st.session_state.turn)
    )
    if uploaded_file is not None:
        image_to_process = uploaded_file


analyze_clicked = st.button(
    "Analyze Board 🤖",
    key="analyze_" + str(st.session_state.game_id) + "_" + str(st.session_state.turn)
)

# Analyse only when the user clicks the button.
if image_to_process is not None and analyze_clicked:
    try:
        image_to_process.seek(0)
        image = Image.open(image_to_process).convert("RGB")
        st.image(image, caption="Image you uploaded", width="stretch")

        with st.spinner("Loading AI models and analysing the board..."):
            yolo_model, tm_model = load_all_models()
            print("DEBUG 7: Starting YOLO inference", flush=True)
            results = yolo_model(image, conf=0.1, verbose=False)
            print("DEBUG 8: YOLO inference completed", flush=True)
        # Get the nine cell boxes found by YOLO.
        if results[0].boxes is None:
            boxes = []
        else:
            boxes = list(results[0].boxes.xyxy.cpu().numpy())

        if len(boxes) != 9:
            st.warning(
                "TicTacBot detected " + str(len(boxes)) +
                " cells instead of 9. Try a clearer photo with the whole board visible."
            )
        else:
            # Sort all cells by their vertical centre first.
            boxes = sorted(boxes, key=sort_by_y)

            # Sort each group of three from left to right.
            first_row = sorted(boxes[0:3], key=sort_by_x)
            second_row = sorted(boxes[3:6], key=sort_by_x)
            third_row = sorted(boxes[6:9], key=sort_by_x)
            boxes = first_row + second_row + third_row

            # Start with an empty 3-by-3 board.
            board = [
                [".", ".", "."],
                [".", ".", "."],
                [".", ".", "."]
            ]

            cell_index = 0
            valid_board = True

            # Crop each cell and ask Teachable Machine what it contains.
            for box in boxes:
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(image.width, int(box[2]))
                y2 = min(image.height, int(box[3]))

                if x2 <= x1 or y2 <= y1:
                    st.error("A detected cell has invalid coordinates. Try another photo.")
                    valid_board = False
                    break

                cell_image = image.crop((x1, y1, x2, y2))
                cell_image.save(temporary_cell_path)
                print("DEBUG 9: Classifying cell " + str(cell_index + 1), flush=True)
                result = tm_model.classify_image(temporary_cell_path)
                print("DEBUG 10: Cell classified " + str(cell_index + 1), flush=True)
                class_index = int(result["class_index"])

                # Convert the cell number into its row and column.
                row = cell_index // 3
                col = cell_index % 3

                # labels.txt must use 0=white_cross, 1=white_circle, 2=empty_cell.
                if class_index == 0:
                    board[row][col] = "X"
                elif class_index == 1:
                    board[row][col] = "O"
                elif class_index == 2:
                    board[row][col] = "."
                else:
                    st.error("Unknown class index. Check the order of labels.txt.")
                    valid_board = False
                    break

                cell_index = cell_index + 1

            if valid_board:
                st.toast("Board scanned successfully!", icon="🎯")
                st.subheader("Detected Board")

                # Display the detected first row.
                c1, c2, c3 = st.columns(3)
                with c1:
                    if board[0][0] == "X":
                        st.info("❌ X")
                    elif board[0][0] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c2:
                    if board[0][1] == "X":
                        st.info("❌ X")
                    elif board[0][1] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c3:
                    if board[0][2] == "X":
                        st.info("❌ X")
                    elif board[0][2] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")

                # Display the detected second row.
                c4, c5, c6 = st.columns(3)
                with c4:
                    if board[1][0] == "X":
                        st.info("❌ X")
                    elif board[1][0] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c5:
                    if board[1][1] == "X":
                        st.info("❌ X")
                    elif board[1][1] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c6:
                    if board[1][2] == "X":
                        st.info("❌ X")
                    elif board[1][2] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")

                # Display the detected third row.
                c7, c8, c9 = st.columns(3)
                with c7:
                    if board[2][0] == "X":
                        st.info("❌ X")
                    elif board[2][0] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c8:
                    if board[2][1] == "X":
                        st.info("❌ X")
                    elif board[2][1] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")
                with c9:
                    if board[2][2] == "X":
                        st.info("❌ X")
                    elif board[2][2] == "O":
                        st.warning("⭕ O")
                    else:
                        st.write("⚫ Empty")

                # Convert the board to the string format used by xo.
                board_string = (
                    board[0][0] + board[0][1] + board[0][2] +
                    board[1][0] + board[1][1] + board[1][2] +
                    board[2][0] + board[2][1] + board[2][2]
                )

                # Use xo to check the game and choose O's move.
                from xo import ai, arbiter
                from xo.board import Board

                game_board = Board.fromstring(board_string.lower())
                game_status = arbiter.outcome(game_board, "o")

                if game_status["status"] == "invalid":
                    st.warning(
                        "This board is not a valid game position. "
                        "Check whether the symbols were detected correctly."
                    )

                elif game_status["status"] == "gameover":
                    if game_status["reason"] == "winner":
                        st.info("Game over! O has already won.")
                    elif game_status["reason"] == "loser":
                        st.info("Game over! X has already won.")
                    else:
                        st.info("Game over! The board is full. It is a draw.")

                else:
                    x_count = game_status["piece_counts"]["xs"]
                    o_count = game_status["piece_counts"]["os"]

                    # The human plays X first, so X must have one more mark than O.
                    if x_count != o_count + 1:
                        st.warning(
                            "Make your move as X, then scan the board again. "
                            "The AI moves only after X has made a move."
                        )
                    else:
                        with st.spinner("The xo AI is choosing its move..."):
                            ai_result = ai.evaluate(game_board, "o")

                        if len(ai_result.positions) == 0:
                            st.warning("The AI did not return a move. Please scan the board again.")
                        else:
                            move = ai_result.positions[0]
                            ai_row = move[0]
                            ai_col = move[1]

                            # xo gives row and column numbers from 1 to 3.
                            if ai_row < 1 or ai_row > 3 or ai_col < 1 or ai_col > 3:
                                st.error("The AI returned a position outside the board.")
                            elif board[ai_row - 1][ai_col - 1] != ".":
                                st.error(
                                    "The AI selected an occupied cell. "
                                    "Please scan the board again."
                                )
                            else:
                                board[ai_row - 1][ai_col - 1] = "O"

                                st.subheader(
                                    "🤖 AI played O at Row " + str(ai_row) +
                                    ", Column " + str(ai_col)
                                )
                                st.success("Updated Board — copy O's move to your physical board!")

                                # Display the updated first row.
                                d1, d2, d3 = st.columns(3)
                                with d1:
                                    if board[0][0] == "X":
                                        st.info("❌ X")
                                    elif board[0][0] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d2:
                                    if board[0][1] == "X":
                                        st.info("❌ X")
                                    elif board[0][1] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d3:
                                    if board[0][2] == "X":
                                        st.info("❌ X")
                                    elif board[0][2] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")

                                # Display the updated second row.
                                d4, d5, d6 = st.columns(3)
                                with d4:
                                    if board[1][0] == "X":
                                        st.info("❌ X")
                                    elif board[1][0] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d5:
                                    if board[1][1] == "X":
                                        st.info("❌ X")
                                    elif board[1][1] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d6:
                                    if board[1][2] == "X":
                                        st.info("❌ X")
                                    elif board[1][2] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")

                                # Display the updated third row.
                                d7, d8, d9 = st.columns(3)
                                with d7:
                                    if board[2][0] == "X":
                                        st.info("❌ X")
                                    elif board[2][0] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d8:
                                    if board[2][1] == "X":
                                        st.info("❌ X")
                                    elif board[2][1] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")
                                with d9:
                                    if board[2][2] == "X":
                                        st.info("❌ X")
                                    elif board[2][2] == "O":
                                        st.warning("⭕ O")
                                    else:
                                        st.write("⚫ Empty")

                                # Check if the AI's move ended the game.
                                updated_string = (
                                    board[0][0] + board[0][1] + board[0][2] +
                                    board[1][0] + board[1][1] + board[1][2] +
                                    board[2][0] + board[2][1] + board[2][2]
                                )
                                updated_board = Board.fromstring(updated_string.lower())
                                updated_status = arbiter.outcome(updated_board, "o")

                                if updated_status["status"] == "gameover":
                                    if updated_status["reason"] == "winner":
                                        st.balloons()
                                        st.success("O has won! Game over.")
                                    elif updated_status["reason"] == "squashed":
                                        st.info("The board is full. The game is a draw.")
                                    else:
                                        st.info("Game over.")
                                else:
                                    st.write(
                                        "Place O on your physical board. Then make your next "
                                        "move as X and scan the board again."
                                    )
                                    if st.button("Next Turn ➔"):
                                        st.session_state.turn += 1
                                        st.rerun()

    except Exception as e:
        st.error("TicTacBot ran into an error while analysing the image.")
        st.write("Error details:", str(e))
        st.info(
            "Check that best.pt, keras_model.h5, and labels.txt are in the same "
            "folder as app.py, and that the packages in requirements.txt are installed."
        )

st.divider()
if st.button("Reset Game"):
    st.session_state.turn = 1
    st.session_state.game_id += 1
    st.rerun()
