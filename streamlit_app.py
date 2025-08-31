import streamlit as st
from snake_ladder import SnakeLadderGame

st.set_page_config(page_title="Snake and Ladder", page_icon="🎲", layout="centered")
st.title("🎲 Snake and Ladder Game")

# Initialize game in session state
if 'game' not in st.session_state:
    st.session_state.game = SnakeLadderGame()
if 'last_roll' not in st.session_state:
    st.session_state.last_roll = None

game = st.session_state.game

# Show board (simple 10x10 grid with emojis)
def draw_board(positions, snakes, ladders):
    snake_emoji = "🐍"
    ladder_emoji = "🪜"
    player1_emoji = "🔴"
    player2_emoji = "🔵"
    both_players_emoji = "🟣"
    board = ""
    for row in range(10, 0, -1):
        for col in range(1, 11):
            num = (row - 1) * 10 + col
            cell = ""
            if positions[0] == num and positions[1] == num:
                cell = both_players_emoji
            elif positions[0] == num:
                cell = player1_emoji
            elif positions[1] == num:
                cell = player2_emoji
            elif num in snakes:
                cell = snake_emoji
            elif num in ladders:
                cell = ladder_emoji
            else:
                cell = f"{num}"
            board += f"{cell:>4}"
        board += "\n"
    st.text(board)
    st.markdown("**Legend:** 🐍=Snake, 🪜=Ladder, 🔴=Player 1, 🔵=Player 2, 🟣=Both Players")

draw_board(game.get_positions(), game.snakes, game.ladders)

if game.get_winner() is not None:
    st.success(f"Player {game.get_winner() + 1} wins! 🎉")
    if st.button("Restart Game"):
        st.session_state.game = SnakeLadderGame()
        st.session_state.last_roll = None
else:
    st.markdown(f"**Player {game.current_player + 1}'s turn**")
    if st.button("Roll Dice"):
        roll = game.roll_dice()
        st.session_state.last_roll = roll
        game.move(roll)
    if st.session_state.last_roll is not None:
        st.info(f"Last roll: {st.session_state.last_roll}")

    st.write(f"Player 1 position: {game.get_positions()[0]}")
    st.write(f"Player 2 position: {game.get_positions()[1]}")