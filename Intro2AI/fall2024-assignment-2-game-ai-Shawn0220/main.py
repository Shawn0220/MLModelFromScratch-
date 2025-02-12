from ai import minimax_alpha_beta
from constants import PLAYER1_PIECE_COLOR, PLAYER2_PIECE_COLOR, SQUARE_SIZE
from display import Display
from game import Game
import pygame


FPS = 60

def main():
    """
    This is the main function to run the game.
    """

    run = True
    clock = pygame.time.Clock()
    game = Game()
    # import board_configs
    # config = getattr(board_configs, f'board_config{12 + 1}')
    # from board import Board
    # game.board = Board(config)
    display = Display()

    while run:
        clock.tick(FPS)

        if game.turn == PLAYER2_PIECE_COLOR:
            # The AI's turn: Use the Minimax algorithm with Alpha-Beta pruning to make a move.
            value, new_board = minimax_alpha_beta(game.get_board(), 3, float('-inf'), float('inf'), True, game)
            game.ai_move(new_board)

        # Check for a winner, and reset game if there is a winner.
        if game.winner() is not None:
            if game.winner() == PLAYER1_PIECE_COLOR:
                print('\nYou won, Human!')
            elif game.winner() == PLAYER2_PIECE_COLOR:
                print('\nThe AI won!')
            game.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Your (human) turn: Select a piece to move.
                pos = pygame.mouse.get_pos()
                row, col = get_click_position_from_mouse(pos)
                game.select(row, col)

        display.update(game.get_board(), game.get_valid_moves())  # Update the game state, and draw the board.

    pygame.quit()

def get_click_position_from_mouse(pos):
    """
    Converts a mouse click to board position using row and column indices.

    Args:
        pos (tuple): Mouse position (x, y)

    Returns:
        tuple: Row and column indices corresponding to the mouse click position
    """

    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

if __name__ == "__main__":

    main()
