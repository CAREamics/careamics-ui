from pathlib import Path

from bioimageio.spec.model.v0_5 import Author as AuthorModel
from qtpy.QtCore import Qt  # type: ignore
from qtpy.QtGui import QPalette, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from careamics_napari.bmz.author_widget import AuthorWidget
from careamics_napari.widgets import bind


class BMZExportWidget(QDialog):
    """A dialog to get information about the model to export."""

    def __init__(
        self,
        parent: QWidget | None = None,
        cover_image_path: str | Path | None = None,
    ):
        """Initialize the widget.

        Parameters
        ----------
        parent : QWidget
            Parent widget. Default is None.
        careamics_config : BaseConfig
            The configuration for the CAREamics algorithm.
        """
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)  # type: ignore

        self.default_cover = str(cover_image_path)
        self.uploaded_cover = None
        self.authors: list[AuthorModel] = []

        # ui
        # model name
        self.name_textbox = QLineEdit()
        self.name_textbox.setPlaceholderText("At least 5 characters")

        # general description
        self.description_textbox = QPlainTextEdit()
        self.description_textbox.setFixedHeight(90)
        self.description_textbox.setPlaceholderText(
            "Enter a short description of your model."
        )

        # data description
        self.data_description_textbox = QPlainTextEdit()
        self.data_description_textbox.setFixedHeight(120)
        self.data_description_textbox.setPlaceholderText(
            "Describe the data that you used for training the model.\n"
            "Use Markdown formatting and must include a 'Validation' header "
            "(like ### Validation) describing validation metrics & methods."
        )

        # authors
        self.authors_listview = QListWidget()
        self.authors_listview.setFixedHeight(90)
        self.add_author_button = QPushButton("Add")
        self.add_author_button.clicked.connect(self._open_new_author)
        self.edit_author_button = QPushButton("Edit")
        self.edit_author_button.clicked.connect(self._open_edit_author)
        self.del_author_button = QPushButton("Remove")
        self.del_author_button.clicked.connect(self._del_author)

        # cover image
        self.image_label = QLabel()
        self.image_label.setBackgroundRole(QPalette.Dark)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)

        self.image_scroll = QScrollArea()
        self.image_scroll.setBackgroundRole(QPalette.Dark)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setFixedHeight(300)
        self.image_scroll.setStyleSheet("border: 1px solid grey;")

        self.default_cover_button = QPushButton("Default Cover Image")
        self.default_cover_button.setMaximumWidth(150)
        self.default_cover_button.clicked.connect(self._set_default_cover_image)
        self.upload_image_button = QPushButton("Upload Cover Image")
        self.upload_image_button.setMaximumWidth(150)
        self.upload_image_button.clicked.connect(self._upload_cover_image)

        # submit & cancel buttons
        self.submit_button = QPushButton("&Export")
        self.submit_button.setMaximumWidth(120)
        self.submit_button.clicked.connect(self._submit)
        self.cancel_button = QPushButton("&Cancel")
        self.cancel_button.setMaximumWidth(120)
        self.cancel_button.clicked.connect(lambda: self.reject())  # type: ignore

        # layout
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        form.addRow("Model Name: ", self.name_textbox)
        form.addRow("General Description:", self.description_textbox)
        form.addRow("Data Description:", self.data_description_textbox)
        vbox = QVBoxLayout()
        vbox.addWidget(self.add_author_button)
        vbox.addWidget(self.edit_author_button)
        vbox.addWidget(self.del_author_button)
        hbox = QHBoxLayout()
        hbox.addWidget(self.authors_listview)
        hbox.addLayout(vbox)
        form.addRow("Authors:", hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.default_cover_button)  # type: ignore
        hbox.addWidget(self.upload_image_button)  # type: ignore
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_scroll)
        vbox.addLayout(hbox)
        vbox.addSpacing(15)
        form.addRow("Cover Image:", vbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.submit_button)
        hbox.addWidget(self.cancel_button)
        form.addRow(hbox)

        self.setLayout(form)
        self.setWindowTitle("BMZ Export")

        self._bind_properties()
        self._set_default_cover_image()

    def _bind_properties(self) -> None:
        """Bind the properties to the UI elements."""
        type(self).model_name = bind(self.name_textbox, "text")
        type(self).general_description = bind(self.description_textbox, "plainText")
        type(self).data_description = bind(self.data_description_textbox, "plainText")
        type(self).cover_image = property(
            fget=lambda self: self.uploaded_cover
            if self.uploaded_cover is not None
            else self.default_cover,
        )

    def _populate_authors_list(self) -> None:
        """Populates the authors' listview widget with the list of authors."""
        self.authors_listview.clear()
        self.authors_listview.addItems(author.name for author in self.authors)

    def _open_new_author(self) -> None:
        """Show author's form to add a new author."""
        author_win = AuthorWidget(self)
        author_win.setWindowModality(Qt.ApplicationModal)  # type: ignore
        author_win.submit.connect(self._add_author)
        author_win.show()

    def _open_edit_author(self) -> None:
        """Show author's form to modify an existing author."""
        selected_index = self.authors_listview.currentRow()
        if selected_index > -1:
            author = self.authors[selected_index]
            author_win = AuthorWidget(self, author)
            author_win.setWindowModality(Qt.ApplicationModal)  # type: ignore
            author_win.submit.connect(
                lambda author: self._update_author(selected_index, author)
            )
            author_win.show()

    def _del_author(self) -> None:
        """Remove the selected author."""
        selected_index = self.authors_listview.currentRow()
        if selected_index > -1:
            reply, del_row = self._remove_from_listview(
                self.authors_listview,
                "Are you sure you want to remove the selected author?",
            )
            if reply:
                del self.authors[del_row]
                self._populate_authors_list()

    def _add_author(self, author: AuthorModel) -> None:
        """Add a new author to the list."""
        self.authors.append(author)
        self._populate_authors_list()

    def _update_author(self, index: int, author: AuthorModel) -> None:
        """Update the author at the given index with the given data."""
        self.authors[index] = author
        self._populate_authors_list()

    def _remove_from_listview(
        self, list_widget: QListWidget, msg: str | None = None
    ) -> tuple[QMessageBox.StandardButton, int]:
        """Removes the selected item from the given listview widget."""
        curr_row = list_widget.currentRow()
        if curr_row > -1:
            reply = QMessageBox.warning(
                self,
                "CAREamics",
                msg or "Are you sure you want to remove the selected item from the list?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                list_widget.takeItem(curr_row)

        return reply == QMessageBox.Yes, curr_row

    def _set_default_cover_image(self) -> None:
        """Set the default cover image."""
        self.image_label.setPixmap(QPixmap(self.default_cover))
        self.image_label.adjustSize()

    def _upload_cover_image(self) -> None:
        """Upload a cover image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cover Image",
            "",
            "Image Files (*.png *.jpg *.bmp);;All Files (*)",
        )
        if file_path:
            self.uploaded_cover = file_path
            self.image_label.setPixmap(QPixmap(file_path))
            self.image_label.adjustSize()

    def _submit(self) -> None:
        """Submit the entered information."""
        if len(self.model_name.strip()) < 5:
            QMessageBox.critical(
                self, "CAREamics", "Model name must be at least 5 characters long!"
            )
            return
        if not self.general_description.strip():
            QMessageBox.critical(self, "CAREamics", "General description is required!")
            return
        if not self.data_description.strip():
            QMessageBox.critical(self, "CAREamics", "Data description is required!")
            return
        if not self.authors:
            QMessageBox.critical(self, "CAREamics", "At least one author is required!")
            return
        if self.cover_image is None:
            QMessageBox.critical(self, "CAREamics", "A cover image is required!")
            return

        self.accept()


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = BMZExportWidget()
    widget.show()

    sys.exit(app.exec_())
