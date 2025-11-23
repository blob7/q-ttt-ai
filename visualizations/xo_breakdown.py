class PieceBreakdownPlot:
    """Horizontal stacked wins (X/O) for each player and separate draws bar."""

    def __init__(self, ax, p1_name: str, p2_name: str):
        self.ax = ax

        # Cumulative counts
        self.p1_x_total = 0
        self.p1_o_total = 0
        self.p2_x_total = 0
        self.p2_o_total = 0
        self.draws_total = 0

        self.p1_name = p1_name
        self.p2_name = p2_name

        ax.set_xlabel("Count")
        ax.set_ylabel("Outcome")
        ax.set_title("Wins by Piece + Draws")

    def update(
        self,
        p1_won_as_x: bool,
        p1_won_as_o: bool,
        p2_won_as_x: bool,
        p2_won_as_o: bool,
        was_draw: bool,
    ):
        # Update internal totals
        self.p1_x_total += int(p1_won_as_x)
        self.p1_o_total += int(p1_won_as_o)
        self.p2_x_total += int(p2_won_as_x)
        self.p2_o_total += int(p2_won_as_o)
        self.draws_total += int(was_draw)

        # Clear previous plot
        self.ax.clear()
        self.ax.set_xlabel("Count")
        self.ax.set_ylabel("Outcome")
        self.ax.set_title("Wins by Piece + Draws")

        # Values and labels
        y_labels = [self.p1_name, self.p2_name, "Draws"]
        y_positions = range(len(y_labels))
        width = 0.4
        max_total = max(
            self.p1_x_total + self.p1_o_total,
            self.p2_x_total + self.p2_o_total,
            self.draws_total,
        )
        color_x = '#1f77b4'
        color_o = '#ff7f0e'
        color_draw = 'gray'

        # Bars
        # Player 1 stacked X/O
        bar_p1_x = self.ax.barh(
            y_positions[0],
            self.p1_x_total,
            height=width,
            color=color_x,
            edgecolor='black',
            label='X Wins'
        )
        bar_p1_o = self.ax.barh(
            y_positions[0],
            self.p1_o_total,
            height=width,
            left=self.p1_x_total,
            color=color_o,
            edgecolor='black',
            label='O Wins'
        )
        # Player 2 stacked X/O
        bar_p2_x = self.ax.barh(
            y_positions[1],
            self.p2_x_total,
            height=width,
            color=color_x,
            edgecolor='black'
        )
        bar_p2_o = self.ax.barh(
            y_positions[1],
            self.p2_o_total,
            height=width,
            left=self.p2_x_total,
            color=color_o,
            edgecolor='black'
        )
        # Draws single bar
        bar_draws = self.ax.barh(
            y_positions[2],
            self.draws_total,
            height=width,
            color=color_draw,
            edgecolor='black',
            label="Draws"
        )

        # Annotate bars
        for x_val, o_val, y_pos in [
            (self.p1_x_total, self.p1_o_total, y_positions[0]),
            (self.p2_x_total, self.p2_o_total, y_positions[1]),
        ]:
            # X segment
            if x_val > 0:
                self.ax.text(
                    x_val / 2, y_pos, str(x_val),
                    ha='center', va='center', color='white', fontweight='bold'
                )
            # O segment
            if o_val > 0:
                self.ax.text(
                    x_val + o_val / 2, y_pos, str(o_val),
                    ha='center', va='center', color='white', fontweight='bold'
                )
        # Draws annotation
        if self.draws_total > 0:
            self.ax.text(
                self.draws_total / 2, y_positions[2], str(self.draws_total),
                ha='center', va='center', color='white', fontweight='bold'
            )

        self.ax.set_yticks(y_positions)
        self.ax.set_yticklabels(y_labels)
        if max_total > 0:
            self.ax.set_xlim(0, max_total * 1.1)
        else:
            self.ax.set_xlim(0, 1)

        # Build legend with unique handles
        handles = [bar_p1_x[0], bar_p1_o[0], bar_draws[0]]
        labels = ['X Wins', 'O Wins', 'Draws']
        self.ax.legend(handles, labels)
