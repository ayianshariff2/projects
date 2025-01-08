public enum Player {
    PLAYER_X('X'),  // Player X symbol
    PLAYER_O('O');  // Player O symbol

    private final char symbol;

    Player(char symbol) {
        this.symbol = symbol;
    }

    public char getSymbol() {
        return symbol;
    }

    // Return the opponent of the current player
    public Player getOpponent() {
        return (this == PLAYER_X) ? PLAYER_O : PLAYER_X;
    }

    @Override
    public String toString() {
        return this == PLAYER_X ? "Player X" : "Player O";
    }
}
