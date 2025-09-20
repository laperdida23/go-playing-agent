import os.path as os
from copy import deepcopy

# import cProfile
# import pstats

baseLocation = os.dirname(os.abspath(__file__))

STEP_FILE = os.join(baseLocation, 'step.txt')

BOARD_SIZE = 5
EMPTY = 0
BLACK = 1
WHITE = 2
KOMI = 2.5

friendlyPieceVal = 1
opposingPieceVal = 1

step = 4 # Intialising in case of errors

minimaxDepth = 3
midGameStep = 9 # Step at which we transition to midgame
endGameStep = 19 # Step at which we transition to endgame

maxStepNumber = 24
CARDINALDIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
surroundingDirections = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

forwardNearbyStones = [(0, 1), (1, 0), (1, 1)]
forwardDiagonalStones = [(1, 1), (1, -1)]

indices = [(0,0), (0,1), (0,2), (0,3), (0,4), 
            (1,0), (1,1), (1,2), (1,3), (1,4), 
            (2,0), (2,1), (2,2), (2,3), (2,4), 
            (3,0), (3,1), (3,2), (3,3), (3,4), 
            (4,0), (4,1), (4,2), (4,3), (4,4)]

centralTerm = set([1, 2, 3])
edgeTerms = set([0, 4])

usefulIndices = [
    (2,2), (2,1), (2,3), (1,2), (3,2),
    (1,1), (1,3), (3,1), (3,3), (0,2), 
    (4,2), (4,1), (4,3), (0,1), (0,3),
    (2,0), (2,4), (1,0), (3,0), (1,4),
    (3,4), (0,0), (4,0), (0,4), (4,4)
]

positionalHeuristics = [
    [0.1, 0.2, 0.2, 0.2, 0.1],
    [0.2, 0.25, 0.25, 0.2, 0.15],
    [0.2, 0.25, 0.3, 0.2, 0.15],
    [0.2, 0.25, 0.2, 0.2, 0.15],
    [0.1, 0.15, 0.15, 0.15, 0.1]
]

midgameHeuristics = [
    [0.1, 0.1, 0.1, 0.1, 0],
    [0.1, 0.2, 0.2, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.2, 0.2, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0]
]

endgameHeuristics = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]
]

def readInput(n, file="input.txt"):
    path = os.join(baseLocation, file)
    with open(path, 'r') as f:
        lines = f.readlines()

        playerColor = int(lines[0])

        olderBoard = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        ongoingBoard = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return playerColor, olderBoard, ongoingBoard


def writeOutput(result, file="output.txt"):
    path = os.join(baseLocation, file)
    with open(path, 'w') as f:
        if result == "PASS":
            f.write("PASS")
        else:
            res = f"{result[0]},{result[1]}"
            f.write(res)

def getStepNumber(playerColor, previousBoard, currentBoard):
    if not os.exists(STEP_FILE):
        with open(STEP_FILE, 'w') as f:
            f.write(str(playerColor+2))
        return playerColor
    else:
        secondMove = False
        firstMove = False

        for i, j in indices:
            if previousBoard[i][j] != EMPTY:
                secondMove = True
                break
            elif currentBoard[i][j] != EMPTY:
                firstMove = True

        if secondMove:
            with open(STEP_FILE, 'r') as f:
                step = int(f.readline())
        else:
            if firstMove:
                step = 2
            else:
                step = 1

        with open(STEP_FILE, 'w') as f:
                f.write(str(step+2))   

        return step

def checkLiberty(board, x, y):
    '''Find if the stone placed at x,y has a liberty.

    :param board: the current playing board.
    :param x: row number of the board.
    :param y: column number of the board.
    :return: boolean indicating whether the given stone still has liberty.
    '''

    for dx, dy in CARDINALDIRECTIONS:
        if 0 <= x+dx < BOARD_SIZE and 0 <= y+dy < BOARD_SIZE:
            if not board[x+dx][y+dy]:
                return True
    return False

def getAlly(board, x, y):
    '''Detect the neighbor allies of a given stone.
    
    :param board: the current playing board.
    :param x: row number of the board.
    :param y: column number of the board.
    :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
    '''

    allVisited, confirmedAlly = set(), []
    ongoingStack = [(x, y)]

    while ongoingStack:
        currX, currY = ongoingStack.pop()
        if (currX, currY) in allVisited:
            continue

        allVisited.add((currX, currY))
        confirmedAlly.append((currX, currY))

        for dx, dy in CARDINALDIRECTIONS:
            testX, testY = currX+dx, currY+dy
            if 0 <= testX < BOARD_SIZE and 0 <= testY < BOARD_SIZE:
                if board[testX][testY] == board[x][y]:
                    ongoingStack.append((testX, testY))

    return confirmedAlly

def checkCapture(board, x, y, playerColor):
    """Check if placing a stone at (x, y) captures any opponent stones.
    
    :param board: the current playing board.
    :param x: row number of the board.
    :param y: column number of the board.
    :return: Integer length of the number of pieces captured or 0 if none.
    """

    opponent = WHITE if playerColor == BLACK else BLACK
    
    for dx, dy in CARDINALDIRECTIONS:
        if 0 <= x+dx < BOARD_SIZE and 0 <= y+dy < BOARD_SIZE and board[x+dx][y+dy] == opponent:
            connectedGroup = getAlly(board, x+dx, y+dy)
            groupLiberties = [checkLiberty(board, oneStone[0], oneStone[1]) for oneStone in connectedGroup]

            if not any(groupLiberties):
                return len(connectedGroup)

    return 0

def violatesKoRule(board, previousBoard, x, y):
    """Check if the current board violates the Ko rule when x, y HAS ALREADY BEEN PLAYED.
    
    :param board: the current playing board.
    :param previousBoard: the previous playing board.
    :return: True if the current board violates the Ko rule, False otherwise.
    """

    return board[x][y] == previousBoard[x][y]

def getValidMoves(previousBoard, currentBoard, color):
    """Get all valid moves for the current player.
    
    :param previousBoard: the previous playing board.
    :param currentBoard: the current playing board.
    :param color: the color of the current player.
    :return: a list of all valid moves.
    """

    validMoves = []

    for i, j in usefulIndices:
        if not currentBoard[i][j]:

            currentBoard[i][j] = color
            closeGroup = getAlly(currentBoard, i, j)
            flag = 0
            for stone in closeGroup:
                if checkLiberty(currentBoard, stone[0], stone[1]):
                    validMoves.append((i, j))
                    currentBoard[i][j] = EMPTY    
                    flag = 1
                    break
            if flag:
                continue

            capturedPieces = checkCapture(currentBoard, i, j, color)

            if capturedPieces > 1:
                validMoves.append((i, j))
            
            if capturedPieces == 1:
                if not violatesKoRule(currentBoard, previousBoard, i, j):
                    validMoves.append((i, j))

            currentBoard[i][j] = EMPTY    

    return validMoves

def shapeHeuristic(board, i, j):
    """Heuristic to evaluate the various basic GO shapes."""
    shapeValue, stoneCount = 0, 0
    cutoffStones = 0

    for dx, dy in forwardNearbyStones:
        if 0 <= i+dx < BOARD_SIZE and 0 <= j+dy < BOARD_SIZE:
            if board[i+dx][j+dy] == board[i][j]:
                stoneCount += 1

    for dx, dy in surroundingDirections:
        if 0 <= i+dx < BOARD_SIZE and 0 <= j+dy < BOARD_SIZE:
            if board[i+dx][j+dy] == board[i][j]:
                cutoffStones -= 1
            elif not board[i+dx][j+dy]:
                cutoffStones += 1

    if cutoffStones > 3:    # Getting dicey with being surrounded
        shapeValue -= 1
        if cutoffStones > 5: # Should not be here, too risky
            shapeValue -= 2

    if stoneCount == 0: # One-Point Jumps MUST BE AVOIDED in 5x5 and prevent cutting
        shapeValue -= 1.25

    if stoneCount == 1:
        for dx, dy in forwardDiagonalStones: 
            if 0 <= i+dx < BOARD_SIZE and 0 <= j+dy < BOARD_SIZE:
                if board[i+dx][j+dy] == board[i][j] and board[i+dx][j] != board[i][j] and board[i][j+dy] != board[i][j]:
                    shapeValue += 0.7   # Kosumi is essential for extension of territory
                else:
                    shapeValue += 1 # Nobumi is good for stability

    if stoneCount == 2: # L Shape helps consolidate the stones and adds stability
        shapeValue += 1.5

    if stoneCount == 3: # Dumpling is not good, too much clustering
        shapeValue -= 3

    return shapeValue

def evaluateBoard(board, playerColor):
    score, opponentColor = (KOMI, BLACK) if playerColor == WHITE else (0, WHITE)
    ownLibertyCount, oppLibertyCount = set(), set()
    edgeBoard, centerBoard = 0, 0

    for i, j in indices:
        if board[i][j] == playerColor:
            score += friendlyPieceVal   # Base score for each stone
            score += positionalHeuristics[i][j]  # Positional heuristic
            score += 0.25 * shapeHeuristic(board, i, j)
            if i in edgeTerms or j in edgeTerms: 
                edgeBoard += 1          # Edge detection for our pieces

        elif board[i][j] == opponentColor:
            score -= opposingPieceVal   # Base opponent penalty
            score -= positionalHeuristics[i][j]
            score -= 0.25 * shapeHeuristic(board, i, j)
        else:
            flag = True
            enemyFlag = True

            if i in centralTerm or j in centralTerm:
                centerBoard += 1
            for dx, dy in surroundingDirections:
                if 0 <= i+dx < BOARD_SIZE and 0 <= j+dy < BOARD_SIZE:
                    if board[i+dx][j+dy] == playerColor:
                        ownLibertyCount.add((i+dx, j+dy))
                        enemyFlag = False
                    elif board[i+dx][j+dy] == opponentColor:
                        flag = False
                        oppLibertyCount.add((i+dx, j+dy))
                    else:
                        enemyFlag = False
                        flag = False
            if flag:
                score += 0.3 # Detected Eye as is only surrounded by own stones
            if enemyFlag:
                score -= 0.3 # Detected Eye as is only surrounded by opponent stones

    score += otherHeuristics(len(ownLibertyCount) - len(oppLibertyCount), edgeBoard, centerBoard)

    return score

def otherHeuristics(libertyDiff, edgeBoard, centerBoard):
    """Other heuristics to evaluate the board."""
    additionalPoints = 0
    edgeWastage = edgeBoard * centerBoard   # Wastage of edge stones when center isn't consolidated

    libertyDiff = 10 if libertyDiff > 10 else libertyDiff # Cap the liberty difference, important but not overly so
    libertyDiff = -10 if libertyDiff < -10 else libertyDiff

    additionalPoints += (libertyDiff - edgeWastage) * 0.2

    return additionalPoints

def evaluateFinalBoard(board, playerColor):
    score, opponentColor = (KOMI, BLACK) if playerColor == WHITE else (0, WHITE)

    for i, j in indices:
        if board[i][j] == playerColor:
            score += 1  # Base score for each stone

        elif board[i][j] == opponentColor:
            score -= 1  # Base opponent penalty

    return score

def resolveCapture(board, x, y):
    """Resolve the capture of the opponent's pieces after stone is placed at x, y
    
    :param board: the current playing board.
    :param x: row number of the board.
    :param y: column number of the board.
    :return: New board with the captured pieces removed and replaced with 0.
    """

    opponent = WHITE if board[x][y] == BLACK else BLACK
    finalGroup = []
    for dx, dy in CARDINALDIRECTIONS:
        if 0 <= x+dx < BOARD_SIZE and 0 <= y+dy < BOARD_SIZE and board[x+dx][y+dy] == opponent:
            connectedGroup = getAlly(board, x+dx, y+dy)
            flag = 1    

            for oneStone in connectedGroup:
                groupLiberty = checkLiberty(board, oneStone[0], oneStone[1])
                if not groupLiberty:
                    continue
                else:
                    flag = 0
                    break

            if flag:
                finalGroup.extend(connectedGroup)
                
    return finalGroup


def resolveBoard(currentBoard, x, y):
    """Resolve the current board.
    
    :param currentBoard: the current playing board.
    :param x: row number of the board.
    :param y: column number of the board.
    :return: the resolved board.
    """

    finalGroup = []

    for dx, dy in CARDINALDIRECTIONS:
        if 0 <= x+dx < BOARD_SIZE and 0 <= y+dy < BOARD_SIZE and currentBoard[x+dx][y+dy] == (3 - currentBoard[x][y]):

                connectedGroup = getAlly(currentBoard, x+dx, y+dy)
                flag = 0
                for oneStone in connectedGroup:
                    flag = 0
                    if checkLiberty(currentBoard, oneStone[0], oneStone[1]):
                        break
                    else:
                        flag = 1
                
                if flag:
                    finalGroup.extend(connectedGroup)

    return finalGroup


def minimax(previousBoard, currentBoard, currentPlayer, depth, alpha, beta, maximizingPlayer):
    
    if depth <= 0:
        if step >= endGameStep and step <= maxStepNumber:
            return evaluateFinalBoard(currentBoard, currentPlayer)
        return evaluateBoard(currentBoard, currentPlayer)
    
    if maximizingPlayer:
        maxEval = float('-inf')
        allMoves = getValidMoves(previousBoard, currentBoard, currentPlayer)

        for currentMove in allMoves:            
            newBoard = deepcopy(currentBoard)
            newBoard[currentMove[0]][currentMove[1]] = currentPlayer

            finalGroup = resolveBoard(newBoard, currentMove[0], currentMove[1])
            for oneStone in finalGroup:
                newBoard[oneStone[0]][oneStone[1]] = EMPTY
            # newBoard = resolveBoard(newBoard, currentMove[0], currentMove[1])

            ongoingEval = minimax(currentBoard, newBoard, currentPlayer, depth-1, alpha, beta, False)
            maxEval = max(maxEval, ongoingEval)
            alpha = max(alpha, ongoingEval)
            if beta <= alpha:
                break

        if not allMoves:
            return evaluateFinalBoard(currentBoard, currentPlayer)
        
        return maxEval
    else:
        minEval = float('inf')
        opponentColor = WHITE if currentPlayer == BLACK else BLACK
        allMoves = getValidMoves(previousBoard, currentBoard, opponentColor)
        for currentMove in allMoves:
            
            newBoard = deepcopy(currentBoard)
            newBoard[currentMove[0]][currentMove[1]] = opponentColor

            finalGroup = resolveBoard(newBoard, currentMove[0], currentMove[1])
            for oneStone in finalGroup:
                newBoard[oneStone[0]][oneStone[1]] = EMPTY

            ongoingEval = minimax(currentBoard, newBoard, currentPlayer, depth-1, alpha, beta, True)

            minEval = min(minEval, ongoingEval)

            beta = min(beta, ongoingEval)

            if beta <= alpha:
                break

        if not allMoves:
            return evaluateFinalBoard(currentBoard, currentPlayer)

        return minEval

def minimaxSelection(previousBoard, currentBoard, playerColor):
    """Select the best move for the current player using minimax algorithm.
    
    :param playerColor: the color of the current player.
    :param previousBoard: the previous playing board.
    :param currentBoard: the current playing board.
    :return: the best move for the current player, or the string PASS.
    """

    bestMove = None
    bestScore = float('-inf')
    global positionalHeuristics, minimaxDepth

    validMoves = getValidMoves(previousBoard, currentBoard, playerColor)

    for move in validMoves:
        newBoard = deepcopy(currentBoard)

        newBoard[move[0]][move[1]] = playerColor
        
        finalGroup = resolveBoard(newBoard, move[0], move[1])

        for oneStone in finalGroup:
            newBoard[oneStone[0]][oneStone[1]] = EMPTY
        
        if finalGroup:
            return move
        
        currentScore = minimax(currentBoard, newBoard, playerColor, minimaxDepth, float('-inf'), float('inf'), False)

        if currentScore > bestScore:
            bestScore = currentScore
            bestMove = move

    if not validMoves:
        return "PASS"

    return bestMove if bestMove else "PASS"

def selectMove(previousBoard, currentBoard, playerColor):
    """Select the first move for the current player.
    
    :param playerColor: the color of the current player.
    :param previousBoard: the previous playing board.
    :param currentBoard: the current playing board.
    :return: the best move for the current player, or the string PASS.
    """

    validMoves = getValidMoves(previousBoard, currentBoard, playerColor)
    bestMove = tuple()

    for move in validMoves:
        bestMove = move
        return bestMove

    return bestMove if bestMove else "PASS"

if __name__ == '__main__':

    currentColor, previousBoard, currentBoard = readInput(BOARD_SIZE)
    # print(currentColor, previousBoard, currentBoard)

    step = getStepNumber(currentColor, previousBoard, currentBoard)
    # step = 3

    if step <= 2: 
        if currentBoard[2][2] == EMPTY:
            writeOutput((2,2))
            exit()
        else:
            writeOutput((2,1))
            exit()
    
    # if step < 9:
    #     minimaxDepth = 4
    if step >= midGameStep:
        positionalHeuristics = midgameHeuristics
        
    if step >= endGameStep:
        positionalHeuristics = endgameHeuristics
        minimaxDepth = maxStepNumber - step    

    # print(step, endGameStep, minimaxDepth)

    if currentColor == BLACK and step < endGameStep:
        opposingPieceVal = 1.25
    else:
        friendlyPieceVal = 1.5

    # bestMove = selectMove(previousBoard, currentBoard, currentColor)
    bestMove = minimaxSelection(previousBoard, currentBoard, currentColor)
    writeOutput(bestMove)

'''
    def main():
'''
'''    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('my_player3.prof')'''
