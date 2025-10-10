from bioimageio.spec.model.v0_5 import Author as AuthorModel
from markdown import markdown
from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import napari.utils.notifications as ntf
except ImportError:
    _has_napari = False
else:
    _has_napari = True


class AuthorWidget(QDialog):
    """Author form widget."""

    submit = Signal(AuthorModel, name="submit")

    def __init__(
        self, parent: QWidget | None = None, author: AuthorModel | None = None
    ) -> None:
        super().__init__(parent)

        # author data (for editing mode)
        self.author = author

        # ui
        self.name_txtbox = QLineEdit()
        self.name_txtbox.setToolTip("Name of the author (Cannot have / or \\).")
        self.email_txtbox = QLineEdit()
        self.affiliation_txtbox = QLineEdit()
        self.git_user_txtbox = QLineEdit()
        self.git_user_txtbox.setToolTip("Github username")
        self.orcid_txtbox = QLineEdit()
        self.orcid_txtbox.setToolTip(
            markdown(AuthorModel.model_fields["orcid"].description)  # type: ignore
        )
        # buttons
        self.submit_button = QPushButton("&Submit")
        self.submit_button.setMaximumWidth(120)
        self.submit_button.clicked.connect(self.submit_author)
        self.cancel_button = QPushButton("&Cancel")
        self.cancel_button.setMaximumWidth(120)
        self.cancel_button.clicked.connect(lambda: self.close())  # type: ignore

        # set values for editing mode
        if self.author is not None:
            self.name_txtbox.setText(self.author.name)
            self.email_txtbox.setText(self.author.email)
            self.affiliation_txtbox.setText(self.author.affiliation)
            self.git_user_txtbox.setText(self.author.github_user)
            self.orcid_txtbox.setText(self.author.orcid)

        # layout
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)  # type: ignore
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow  # type: ignore
        )
        form.addRow("Name (required):", self.name_txtbox)
        form.addRow("Email:", self.email_txtbox)
        form.addRow("Affiliation:", self.affiliation_txtbox)
        form.addRow("Github User:", self.git_user_txtbox)
        form.addRow("ORCID:", self.orcid_txtbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.submit_button)
        hbox.addWidget(self.cancel_button)

        vbox = QVBoxLayout()
        vbox.addLayout(form)
        vbox.addSpacing(10)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def submit_author(self) -> None:
        """Validate and submit the entered author's profile."""
        author_data = {
            "name": self._get_value(self.name_txtbox.text()),
            "email": self._get_value(self.email_txtbox.text()),
            "affiliation": self._get_value(self.affiliation_txtbox.text()),
            "github_user": self._get_value(self.git_user_txtbox.text()),
            "orcid": self._get_value(self.orcid_txtbox.text()),
        }
        # validation
        try:
            author = AuthorModel.model_validate(author_data)
        except Exception as err:
            print(err)
            if _has_napari:
                ntf.show_error("Not a valid author!")
            return

        # submit
        self.submit.emit(author)
        self.close()

    def _get_value(self, str) -> str | None:
        if not str.strip():
            return None
        else:
            return str.strip()


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = AuthorWidget()
    widget.show()

    sys.exit(app.exec_())
