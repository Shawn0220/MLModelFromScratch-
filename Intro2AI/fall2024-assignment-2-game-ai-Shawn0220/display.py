from constants import BLUE, DARK_SQUARE_COLOR, LIGHT_SQUARE_COLOR, PLAYER1_PIECE_COLOR, PLAYER2_PIECE_COLOR, ROWS, COLS, SQUARE_SIZE, WIDTH, HEIGHT
import pygame


CROWN = pygame.transform.scale(pygame.image.load('crown.png'), (45, 25))

class Display:
    def __init__(self):
        """
        Initializes the display object by setting a display window and giving it a caption.
        """

        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Checkers')

    def draw_board(self, board):
        """
        Draws the entire board, including the pieces, within the given display window.

        Args:
            board (Board): The board to draw
        """

        self.draw_squares()
        pieces = board.get_all_pieces(PLAYER1_PIECE_COLOR) + board.get_all_pieces(PLAYER2_PIECE_COLOR)
        for piece in pieces:
            if piece != 0:
                self.draw_piece(piece)

    def draw_squares(self):
        """
        Draws the Checkers board squares.
        """

        self.win.fill(DARK_SQUARE_COLOR)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(self.win, LIGHT_SQUARE_COLOR, (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_piece(self, piece):
        """
        Draws the given piece.

        Args:
            piece (Piece): The piece to draw
        """

        padding = 15
        outline = 2
        radius = SQUARE_SIZE // 2 - padding
        x = SQUARE_SIZE * piece.col + SQUARE_SIZE // 2
        y = SQUARE_SIZE * piece.row + SQUARE_SIZE // 2
        pygame.draw.circle(self.win, piece.color, (x, y), radius + outline)
        pygame.draw.circle(self.win, piece.color, (x, y), radius)

        if piece.king:
            self.win.blit(CROWN, (x - CROWN.get_width() // 2, y - CROWN.get_height() // 2))

    def draw_valid_moves(self, valid_moves):
        """
        Draws indicators on the board for the valid moves.

        Args:
            valid_moves (dict): A dictionary where keys are tuples representing target positions (row, col) and values
            are lists of pieces to be captured
        """

        for move in valid_moves:
            row, col = move
            pygame.draw.circle(self.win, BLUE,
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)

    def display_board(self, board):
        """
        Displays the board.

        Args:
            board (Board): The board to display
        """

        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            self.draw_board(board)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

        pygame.quit()

    def update(self, board, valid_moves):
        """
        Updates the display by drawing the board and indicating the valid moves. Then it refreshes the screen.

        Args:
            board (Board): The board to display
            valid_moves (dict): A dictionary where keys are tuples representing target positions (row, col) and values
            are lists of pieces to be captured
        """

        self.draw_board(board)
        self.draw_valid_moves(valid_moves)
        pygame.display.update()