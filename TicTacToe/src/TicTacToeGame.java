import java.util.Scanner;

public class TicTacToeGame {

    private static TicTacToeBoard board;
    private static Player currentPlayer;

    public static void main(String[] args) {
        // Ask the user for the size of the board
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the size of the board (e.g., 3 for a 3x3 grid): ");
        int size = scanner.nextInt();

        // Initialize the game board with the given size and start with player X
        board = new TicTacToeBoard(size);
        currentPlayer = Player.PLAYER_X;

        // Start the game loop
        while (true) {
            // Display the current board
            board.displayBoard();

            // Prompt the current player to make a move
            promptPlayerMove();

            // Check if the game is over
            if (board.isBoardFullOrWon()) {
                board.displayBoard();
                announceGameResult();
                break;
            }

            // Toggle to the other player
            togglePlayer();
        }
    }

    // Prompt the current player to make a move
    private static void promptPlayerMove() {
        Scanner scanner = new Scanner(System.in);
        int row, col;

        while (true) {
            System.out.println(currentPlayer + ", enter your move (row, col). Top left corner is (0,0): ");
            row = scanner.nextInt();
            col = scanner.nextInt();

            if (isMoveValid(row, col)) {
                board.makeMove(row, col, currentPlayer);
                break;
            } else {
                System.out.println("Invalid move. Try again.");
            }
        }
    }

    // Check if the move is valid (within bounds and on an empty space)
    private static boolean isMoveValid(int row, int col) {
        return row >= 0 && row < board.getSize() && col >= 0 && col < board.getSize()
                && board.getSymbolAt(row, col) == TicTacToeBoard.EMPTY;
    }

    // Toggle to the next player
    private static void togglePlayer() {
        currentPlayer = currentPlayer.getOpponent();
    }

    // Announce the result of the game (winner or tie)
    private static void announceGameResult() {
        if (board.checkWinner()) {
            System.out.println(currentPlayer + " wins!");
        } else if (board.isBoardFull()) {
            System.out.println("It's a tie!");
        }
    }
}

