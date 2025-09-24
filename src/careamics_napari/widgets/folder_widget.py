"""A widget used for selecting an existing folder."""

from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget
from typing_extensions import Self


class FolderWidget(QWidget):
    """A widget used for selecting an existing folder.

    Parameters
    ----------
    text : str
        Text displayed on the button.
    """

    def __init__(self: Self, text: str) -> None:
        """Initialize the widget.

        Parameters
        ----------
        text : str
            Text displayed on the button.
        """
        super().__init__()

        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.setContentsMargins(0, 0, 0, 0)

        # text field
        self.text_field = QLineEdit("")
        self.text_field.setReadOnly(True)
        hbox.addWidget(self.text_field)

        # folder selection button
        self.button = QPushButton(text)
        hbox.addWidget(self.button)
        self.button.clicked.connect(self._open_dialog)

        self.setLayout(hbox)

    def _open_dialog(self: Self) -> None:
        """Open a dialog to select a folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(path)

        # set text in the text field
        self.text_field.setText(path)

    def get_folder(self: Self) -> str:
        """Get the selected folder.

        Returns
        -------
        str
            The selected folder as read out from the text field.
        """
        return self.text_field.text()

    def get_text_widget(self: Self) -> QLineEdit:
        """Get the text widget.

        Returns
        -------
        QLineEdit
            The text widget.
        """
        return self.text_field
