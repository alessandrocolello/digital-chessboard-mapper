import random

def isSameColorSquare(row1, col1, row2, col2):
    return (row1 + col1) % 2 == (row2 + col2) % 2

def generateRandomFenWithConstraints(pieces):

    chessboard = [[' ' for _ in range(8)] for _ in range(8)]
    placed_pieces = {}
    
    for piece, count in pieces.items():
        for _ in range(count):
            while True:
                row = random.randint(0, 7)
                col = random.randint(0, 7)
                
                if chessboard[row][col] == ' ':
                    # Check if the piece is a bishop and there's already another bishop placed
                    if piece == 'P' and 'P' in placed_pieces:
                        # Ensure that the bishops are on different color squares
                        other_row, other_col = placed_pieces['P']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue
                    
                    # Check if the piece is a knight and there's already another knight placed
                    if piece == 'B' and 'B' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['B']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue

                    if piece == 'N' and 'N' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['N']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue

                    if piece == 'R' and 'R' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['R']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue

                    # Check if the piece is a bishop and there's already another bishop placed
                    if piece == 'p' and 'p' in placed_pieces:
                        # Ensure that the bishops are on different color squares
                        other_row, other_col = placed_pieces['p']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue
                    
                    # Check if the piece is a knight and there's already another knight placed
                    if piece == 'b' and 'b' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['b']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue

                    if piece == 'n' and 'n' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['n']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue

                    if piece == 'r' and 'r' in placed_pieces:
                        # Ensure that the knights are on different color squares
                        other_row, other_col = placed_pieces['r']
                        if isSameColorSquare(row, col, other_row, other_col):
                            continue
                    
                    # If there are no constraints or the constraints are satisfied, place the piece
                    chessboard[row][col] = piece
                    placed_pieces[piece] = (row, col)
                    break
    
    fen = '/'.join(''.join(row) for row in chessboard)
    fen = fen.replace(' '*8, '8')
    fen = fen.replace(' '*7, '7')
    fen = fen.replace(' '*6, '6')
    fen = fen.replace(' '*5, '5')
    fen = fen.replace(' '*4, '4')
    fen = fen.replace(' '*3, '3')
    fen = fen.replace(' '*2, '2')
    fen = fen.replace(' '*1, '1')
    
    return fen

# Example usage:
pieces = {'P':0, 'B':0, 'N':0, 'R':0, 'Q':1, 'K':1, 'p':0, 'b':0, 'n':0, 'r':0, 'q':1, 'k':1}

print(generateRandomFenWithConstraints(pieces))
