class Connect4:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[' ' for _ in range(cols)] for _ in range(rows)]
        self.current_winner = None

    def print_board(self):
        for row in self.board:
            print('| ' + ' | '.join(row) + ' |')
        print('  ' + '   '.join(str(i) for i in range(self.cols)))

    def available_moves(self):
        moves = []
        for col in range(self.cols):
            if self.board[0][col] == ' ':
                moves.append(col)
        return moves

    def empty_squares(self):
        return len(self.available_moves()) > 0

    def make_move(self, col, letter):
        if self.board[0][col] != ' ':
            return False
        for row in reversed(range(self.rows)):
            if self.board[row][col] == ' ':
                self.board[row][col] = letter
                if self.check_winner(row, col, letter):
                    self.current_winner = letter
                return True
        return False

    def check_winner(self, row, col, letter):
        count = 0
        for c in range(max(0, col-3), min(self.cols, col+4)):
            if self.board[row][c] == letter:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        count = 0
        for r in range(max(0, row-3), min(self.rows, row+4)):
            if self.board[r][col] == letter:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        count = 0
        for d in range(-3, 4):
            r = row + d
            c = col + d
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.board[r][c] == letter:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        count = 0
        for d in range(-3, 4):
            r = row + d
            c = col - d
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.board[r][c] == letter:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        return False