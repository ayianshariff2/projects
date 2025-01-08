public class TicTacToeBoard {

    protected static final char EMPTY = ' ';
    private char[][] grid;
    private int size;

    public TicTacToeBoard(int size) {
        this.size = size;
        grid = new char[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                grid[i][j] = EMPTY;
            }
        }
    }

    // Display the current board on the console
    public void displayBoard() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                System.out.print(grid[i][j]);
                if (j < size - 1) {
                    System.out.print("|");
                }
            }
            System.out.println();
            if (i < size - 1) {
                System.out.println("-".repeat(size * 2 - 1));
            }
        }
    }

    // Make a move on the board
    public void makeMove(int row, int col, Player player) {
        grid[row][col] = player.getSymbol();
    }

    // Check if the game is over (either by a win or the board being full)
    public boolean isBoardFullOrWon() {
        return checkWinner() || isBoardFull();
    }

    // Check if there is a winner on the board
    public boolean checkWinner() {
        for (int i = 0; i < size; i++) {
            if (checkRow(i) || checkColumn(i)) {
                return true;
            }
        }
        return checkDiagonal1() || checkDiagonal2();
    }

    // Check if a specific row has a winning combination
    private boolean checkRow(int row) {
        for (int col = 1; col < size; col++) {
            if (grid[row][col] != grid[row][0] || grid[row][col] == EMPTY) {
                return false;
            }
        }
        return true;
    }

    // Check if a specific column has a winning combination
    private boolean checkColumn(int col) {
        for (int row = 1; row < size; row++) {
            if (grid[row][col] != grid[0][col] || grid[row][col] == EMPTY) {
                return false;
            }
        }
        return true;
    }

    // Check if the first diagonal has a winning combination
    private boolean checkDiagonal1() {
        for (int i = 1; i < size; i++) {
            if (grid[i][i] != grid[0][0] || grid[i][i] == EMPTY) {
                return false;
            }
        }
        return true;
    }

    // Check if the second diagonal has a winning combination
    private boolean checkDiagonal2() {
        for (int i = 1; i < size; i++) {
            if (grid[i][size - 1 - i] != grid[0][size - 1] || grid[i][size - 1 - i] == EMPTY) {
                return false;
            }
        }
        return true;
    }

    // Check if the board is full (no empty spaces)
    public boolean isBoardFull() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (grid[i][j] == EMPTY) {
                    return false;
                }
            }
        }
        return true;
    }

    // Get the symbol at a specific position (for validation)
    public char getSymbolAt(int row, int col) {
        return grid[row][col];
    }

    public int getSize() {
        return size;
    }
}
