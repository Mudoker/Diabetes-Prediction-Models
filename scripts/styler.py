class Styler:
    """
    A class for styling texts.

    Attributes:
        None

    Methods:
        draw_box(text): Draws a box with text centered.
    """

    def draw_box(self, text):
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
        width = len(text) + 4  # Add extra padding on both sides
        height = 3

        # Top border
        print("┌" + "─" * (width) + "┐")

        # Middle row with text
        print("│" + " " * 2 + text + " " * 2 + "│")

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
