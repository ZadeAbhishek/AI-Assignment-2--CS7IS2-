class TicTacToe:
    def __init__(self):
        # A single list of 9 spaces representing the 3x3 board.
        self.board = [' '] * 9
        self.current_winner = None

    def print_board(self):
        # Print board rows along with indices.
        for row_idx in range(3):
            row = self.board[row_idx * 3:(row_idx + 1) * 3]
            indices = [str(i) for i in range(row_idx * 3, (row_idx + 1) * 3)]
            print('| ' + ' | '.join(row) + ' |' + " <- " + ' '.join(indices))
        print()

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square, letter):
        # Check the row.
        row_idx = square // 3
        row = self.board[row_idx * 3:(row_idx + 1) * 3]
        if all(spot == letter for spot in row):
            return True
        
        # Check the column.
        col_idx = square % 3
        column = [self.board[col_idx + i * 3] for i in range(3)]
        if all(spot == letter for spot in column):
            return True
        
        # Check diagonals if the square is even (only even indices can be on a diagonal).
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all(spot == letter for spot in diagonal1):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all(spot == letter for spot in diagonal2):
                return True
        
        return False