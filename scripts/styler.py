class Styler:
    """
    A class for styling texts.

    Attributes:
        None

    Methods:
        draw_box(text): Draws a box with text centered.
    """

    def draw_box(self, text, spacing=2):
        """
        Draws a box with text centered.

        Args:
            text (str): The text to be displayed in the box.

        Returns:
            None
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        # Calculate width and height based on text size
        width = len(text) + spacing + 2  # Add extra padding on both sides

        # Top border
        print("┌" + "─" * (width) + "┐")

        # Middle row with text
        print("│" + " " * spacing + text + " " * spacing + "│")

        # Bottom border
        print("└" + "─" * (width) + "┘")

    def style(self, text, bold=False, italic=False, underline=False):
        """
        Applies specified styles to the given text.

        Args:
            text (str): The text to be styled.
            bold (bool): Whether to apply bold formatting.
            italic (bool): Whether to apply italic formatting.
            underline (bool): Whether to apply underline formatting.

        Returns:
            str: The styled text.
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        formatting = ""
        if bold:
            formatting += "\033[1m"
        if italic:
            formatting += "\033[3m"
        if underline:
            formatting += "\033[4m"

        styled_text = formatting + text + "\033[0m"
        return styled_text

    def to_table(self, data=[], columns=[], index=False, header=True):
        """
        Converts data into a formatted table.

        Args:
            data (List[List]): The data to be converted into a table.
            columns (List): Optional. The column labels of the table.
            index (bool): Optional. Whether to include row indices.
            header (bool): Optional. Whether to include column headers.

        Returns:
            str: The formatted table as a string.
        """

        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("Input data must be a list of lists.")

        if columns and not all(isinstance(col, str) for col in columns):
            raise TypeError("Column names must be strings.")

        if not isinstance(index, bool) or not isinstance(header, bool):
            raise TypeError("Index and header must be boolean values.")

        # Create a table in AsciiDoc format
        table = ""
        if header:
            if columns:
                table += "|===\n| " + " | ".join(columns) + "\n"
            else:
                table += "|===\n"

        for row in data:
            table += "| " + " | ".join(str(cell) for cell in row) + "\n"

        if index:
            table = "[width='100%']\n" + table

        table += "|===\n"
        return table
