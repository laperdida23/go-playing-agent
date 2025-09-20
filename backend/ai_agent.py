"""
Go Playing AI Agent - A Sophisticated 5x5 Go Game AI

This program implements an intelligent Go (Weiqi/Baduk) playing agent designed for 5x5 boards.
The AI uses the minimax algorithm with alpha-beta pruning to play competitively against human 
players or other AI agents.

GAME OVERVIEW:
Go is an ancient strategic board game where two players (Black and White) alternately place
stones on intersections of a grid to control territory. The player who controls more territory
at the end wins.

KEY GO RULES IMPLEMENTED:
1. CAPTURING: Stones with no liberties (empty adjacent spaces) are captured and removed
2. KO RULE  : You cannot immediately recapture a single stone that just captured your stone
3. SUICIDE  : You cannot place a stone that would have no liberties (unless it captures)
4. TERRITORY: Empty spaces surrounded by your stones count as your territory, but is not scored
5. KOMI     : White gets bonus points (2.5) to compensate for Black playing first

GAME CONSTRAINTS:
- Board Size    : 5x5 (25 intersections)
- Maximum Moves : 24 total (12 per player)
- Time Limit    : Must make moves quickly
- Input/Output  : Reads from input.txt, step.txt and writes to output.txt

AI STRATEGY:
1. OPENING: Plays center or near-center for maximum influence
2. MIDGAME: Uses minimax with board evaluation for tactical play
3. ENDGAME: Focuses on territory counting and secure positions

ALGORITHM COMPONENTS:
- Minimax Algorithm: Looks ahead several moves to find best play
- Alpha-Beta Pruning: Optimizes search by eliminating bad branches
- Board Evaluation: Sophisticated scoring considering multiple factors
- Game Phase Detection: Adjusts strategy based on move number
- Heuristic Functions: Evaluates stone shapes, liberty, territory, etc.
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

# import pstats             # Profiling imports for identifying bottlenecks
# import cProfile
import os.path as os        # For file path operations
from copy import deepcopy   # For creating deep copies of game boards

# Get the directory where this script is located (for file operations)
baseLocation = os.dirname(os.abspath(__file__))

# File path for storing the current step/move number
STEP_FILE = os.join(baseLocation, 'step.txt')


# ============================================================================
# GAME CONSTANTS - Core Go Game Rules and Settings
# ============================================================================

BOARD_SIZE = 5              # 5x5 board (25 intersections total)
EMPTY = 0                   # Empty intersection (no stone placed)
BLACK = 1                   # Black stone/player (plays first)
WHITE = 2                   # White stone/player (plays second)
KOMI = 2.5                  # Points given to White to compensate for playing second


# ============================================================================
# AI STRATEGY CONSTANTS - Control AI behavior and strength
# ============================================================================

friendlyPieceVal = 1        # Base value for our own stones
opposingPieceVal = 1        # Base penalty for opponent stones

step = 4                    # Current move number (initialized for error handling)

minimaxDepth = 3            # How many moves ahead the AI looks (deeper = stronger but slower)
midGameStep = 9             # Move number when middle game begins
endGameStep = 19            # Move number when end game begins (territory counting focus)

maxStepNumber = 24          # Maximum possible moves in the game



# ============================================================================
# DIRECTION VECTORS - For checking adjacent positions on the board
# ============================================================================

# Cardinal directions: up, down, left, right (for checking liberties and captures)
CARDINALDIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# All 8 surrounding directions (cardinal + diagonal) - for territory and influence
surroundingDirections = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

# Forward directions for shape analysis (helps evaluate stone connections from corners and otherwise)
forwardNearbyStones = [(0, 1), (1, 0), (1, 1)]
forwardDiagonalStones = [(1, 1), (1, -1)]

# ============================================================================
# BOARD POSITION MANAGEMENT
# ============================================================================

# All possible positions on a 5x5 board (row, column) - for iterating through board
indices = [(0,0), (0,1), (0,2), (0,3), (0,4), 
            (1,0), (1,1), (1,2), (1,3), (1,4), 
            (2,0), (2,1), (2,2), (2,3), (2,4), 
            (3,0), (3,1), (3,2), (3,3), (3,4), 
            (4,0), (4,1), (4,2), (4,3), (4,4)]

# Position classification for strategic evaluation
centralTerm = set([1, 2, 3])                # Central rows/columns (more valuable positions)
edgeTerms = set([0, 4])                     # Edge rows/columns (less valuable, harder to defend)

# Move priority order: try center first, then expand outward (best-first search optimization)
usefulIndices = [
    (2,2), (2,1), (2,3), (1,2), (3,2),      # Center and immediate neighbors
    (1,1), (1,3), (3,1), (3,3), (0,2),      # Secondary center positions  
    (4,2), (4,1), (4,3), (0,1), (0,3),      # Extending toward edges
    (2,0), (2,4), (1,0), (3,0), (1,4),      # Edge positions
    (3,4), (0,0), (4,0), (0,4), (4,4)       # Corner positions (lowest priority)
]

# ============================================================================
# POSITIONAL HEURISTICS - Strategic value tables for different game phases
# ============================================================================

# OPENING GAME: Emphasize center control and influence
# Higher values = more desirable positions to play
positionalHeuristics = [
    [0.1, 0.2, 0.2, 0.2, 0.1],              # Top row: edges less valuable
    [0.2, 0.25, 0.25, 0.2, 0.15],           # Second row: good but not center
    [0.2, 0.25, 0.3, 0.2, 0.15],            # Middle row: center (2,2) most valuable
    [0.2, 0.25, 0.2, 0.2, 0.15],            # Fourth row: symmetric to second
    [0.1, 0.15, 0.15, 0.15, 0.1]            # Bottom row: edges least valuable
]

# MIDDLE GAME: Maintain center focus but reduce edge penalties
midgameHeuristics = [
    [0.1, 0.1, 0.1, 0.1, 0],                # Avoid top-right corner specifically
    [0.1, 0.2, 0.2, 0.2, 0.1],              # Consistent value across
    [0.1, 0.2, 0.3, 0.2, 0.1],              # Still prefer center
    [0.1, 0.2, 0.2, 0.2, 0.1],              # Symmetric play
    [0.1, 0.1, 0.1, 0.1, 0]                 # Avoid bottom-right corner
]

# END GAME: All positions equal - focus on stone count and captures only
endgameHeuristics = [
    [0.0, 0.0, 0.0, 0.0, 0.0],              # No positional preference
    [0.0, 0.0, 0.0, 0.0, 0.0],              # Pure tactical evaluation
    [0.0, 0.0, 0.0, 0.0, 0.0],              # Territory counting dominates
    [0.0, 0.0, 0.0, 0.0, 0.0],              # No strategic biases
    [0.0, 0.0, 0.0, 0.0, 0.0]               # Let minimax handle everything
]

# ============================================================================
# INPUT/OUTPUT FUNCTIONS - File handling for game communication
# ============================================================================

def readInput(n, file="input.txt"):
    """
    Reads the current game state from an input file.
    
    The input file format is:
    Line 1      : Player color (1 for Black, 2 for White)
    Lines 2-6   : Previous board state (5x5 grid of numbers)
    Lines 7-11  : Current board state (5x5 grid of numbers)
    
    Each board position contains:
    0 = Empty intersection
    1 = Black stone
    2 = White stone
    
    Args:
        n (int)     : Board size (should be 5 for our 5x5 game)
        file (str)  : Name of input file (default: "input.txt")
    
    Returns:
        tuple: (playerColor, olderBoard, ongoingBoard)
        - playerColor (int)     : 1 for Black, 2 for White
        - olderBoard (list)     : 5x5 list representing previous game state
        - ongoingBoard (list)   : 5x5 list representing current game state
    
    Example:
        If input.txt contains:
        1
        00000
        00000
        00100
        00000
        00000
        00000
        00000
        00120
        00000
        00000
        
        This means: Black to play, previous board had Black stone at (2,2),
        current board has Black at (2,2) and White at (2,3)
    """
    path = os.join(baseLocation, file)
    with open(path, 'r') as f:
        lines = f.readlines()

        # Read player color from first line
        playerColor = int(lines[0])

        # Read previous board state (lines 1 to n)
        olderBoard = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        
        # Read current board state (lines n+1 to 2n)
        ongoingBoard = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return playerColor, olderBoard, ongoingBoard

def writeOutput(result, file="output.txt"):
    """
    Writes the AI's chosen move to an output file.
    
    The output format is either:
    - "PASS" if no good move is available
    - "row,column" for the chosen move (e.g., "2,3" for row 2, column 3)
    
    Args:
        result      : Either the string "PASS" or a tuple (row, col) representing the move
        file (str)  : Name of output file (default: "output.txt")
    
    Examples:
        writeOutput("PASS") -> writes "PASS" to output.txt
        writeOutput((2, 3)) -> writes "2,3" to output.txt
    """
    path = os.join(baseLocation, file)
    with open(path, 'w') as f:
        if result == "PASS":
            f.write("PASS")
        else:
            # Convert move tuple to comma-separated string
            res = f"{result[0]},{result[1]}"
            f.write(res)

def getStepNumber(playerColor, previousBoard, currentBoard):
    """
    Determines the current move number in the game.
    
    This function tracks game progress by:
    1. Checking if step.txt exists (contains move counter)
    2. If not, determines if this is move 1 or 2 based on board state
    3. Updates the step file with next move number
    
    Move numbering:
    - Move 1: First Black stone
    - Move 2: First White stone  
    - Move 3: Second Black stone
    - etc.
    
    Args:
        playerColor (int)   : Current player (1=Black, 2=White)
        previousBoard (list): Previous board state (5x5)
        currentBoard (list) : Current board state (5x5)
    
    Returns:
        int: Current move number (1-24)
        
    Note:
        This function has side effects - it reads and writes to step.txt
        to maintain game state between moves. If step does not exist after 2 moves
        have passed, it will malfunction in the count
    """
    if not os.exists(STEP_FILE):
        # First move of the game - create step file
        with open(STEP_FILE, 'w') as f:
            f.write(str(playerColor+2))  # Write 3 for Black's first move, 4 for White's
        return playerColor
    else:
        # Game is in progress - determine current move number
        secondMove = False  # Has the second move been made?
        firstMove = False   # Has the first move been made?

        # Check if there are any stones on previous board (indicating moves have been made)
        for i, j in indices:
            if previousBoard[i][j] != EMPTY:
                secondMove = True  # At least one move has been made previously
                break
            elif currentBoard[i][j] != EMPTY:
                firstMove = True   # Current board has moves but previous doesn't

        if secondMove:
            # Game has been going on - read step number from file
            with open(STEP_FILE, 'r') as f:
                step = int(f.readline())
        else:
            # Early game - determine move number based on board state
            if firstMove:
                step = 2  # Second move of game (first White move)
            else:
                step = 1  # First move of game (first Black move)

        # Update step file with next move number (increment by 2 since it alternates players)
        with open(STEP_FILE, 'w') as f:
                f.write(str(step+2))   

        return step

# ============================================================================
# CORE GO GAME LOGIC - Liberty, Capture, and Rule Checking
# ============================================================================

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
