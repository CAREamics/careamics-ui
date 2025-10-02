"""A widget displaying losses and a button to open TensorBoard in the browser."""

import webbrowser
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from magicgui.widgets import Container
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QCursor, QIcon, QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from typing_extensions import Self

from careamics_napari.resources import ICON_TF


# TODO why is it a magicgui container and not just a widget?
class TBPlotWidget(Container):
    """A widget displaying losses and a button to open TensorBoard in the browser.

    Parameters
    ----------
    min_width : int or None, default=None
        Minimum width of the widget.
    min_height : int or None, default=None
        Minimum height of the widget.
    max_width : int or None, default=None
        Maximum width of the widget.
    max_height : int or None, default=None
        Maximum height of the widget.
    """

    # TODO what is this method used for?
    def __setitem__(self: Self, key: Any, value: Any) -> None:
        """Ignore set item.

        Parameters
        ----------
        key : Any
            Ignored.
        value : Any
            Ignored.
        """
        pass

    def __init__(
        self: Self,
        min_width: int | None = None,
        min_height: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        work_dir: Path | None = None,
    ):
        """Initialize the widget.

        Parameters
        ----------
        min_width : int or None, default=None
            Minimum width of the widget.
        min_height : int or None, default=None
            Minimum height of the widget.
        max_width : int or None, default=None
            Maximum width of the widget.
        max_height : int or None, default=None
            Maximum height of the widget.
        """
        super().__init__()

        # self.train_signal = train_signal
        self.work_dir = work_dir

        if max_width:
            self.native.setMaximumWidth(max_width)
        if max_height:
            self.native.setMaximumHeight(max_height)
        if min_width:
            self.native.setMinimumWidth(min_width)
        if min_height:
            self.native.setMinimumHeight(min_height)

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(None)
        self.native.layout().addWidget(self.graphics_widget)

        # plot widget
        self.plot = self.graphics_widget.addPlot()
        self.plot.setLabel("bottom", "epoch")
        self.plot.setLabel("left", "loss")
        self.plot.addLegend(offset=(125, -50))

        # tensorboard button
        tb_button = QPushButton("Open in TensorBoard")
        tb_button.setToolTip("Open TensorBoard in your browser")
        tb_button.setIcon(QIcon(QPixmap(ICON_TF)))
        tb_button.setLayoutDirection(Qt.LeftToRight)  # type: ignore
        tb_button.setIconSize(QSize(32, 29))
        tb_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        tb_button.clicked.connect(self.open_tb)

        # add to layout on the bottom left
        hbox = QHBoxLayout()
        hbox.addWidget(tb_button)
        hbox.addWidget(QLabel(""))
        button_widget = QWidget()
        button_widget.setLayout(hbox)
        self.native.layout().addWidget(button_widget)

        # set empty references
        self.epochs: list[int] = []
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.url: str | None = None
        self.tb = None

    def stop_tb(self: Self) -> None:
        """Stop the TensorBoard process.

        Currently not implemented.
        """
        # haven't found any good way to stop the tb process, there's currently no API
        # for it
        pass

    def open_tb(self: Self) -> None:
        """Open TensorBoard in the browser."""
        if self.tb is None and self.work_dir is not None:
            from tensorboard import program

            self.tb = program.TensorBoard()

            path = str(self.work_dir / "logs" / "lightning_logs")
            self.tb.configure(argv=[None, "--logdir", path])  # type: ignore
            self.url = self.tb.launch()  # type: ignore

            if self.url is not None:
                webbrowser.open(self.url)
        else:
            if self.url is not None:
                webbrowser.open(self.url)

    def update_plot(self: Self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Update the plot with new data.

        Parameters
        ----------
        epoch : int
            Epoch number.
        train_loss : float
            Training loss.
        val_loss : float
            Validation loss.
        """
        # clear the plot
        self.plot.clear()

        # add the new points
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        # replot
        self.plot.plot(
            self.epochs,
            self.train_loss,
            pen=pg.mkPen(color=(204, 221, 255)),
            symbol="o",
            symbolSize=2,
            name="Train",
        )
        self.plot.plot(
            self.epochs,
            self.val_loss,
            pen=pg.mkPen(color=(244, 173, 173)),
            symbol="o",
            symbolSize=2,
            name="Val",
        )

    def clear_plot(self: Self) -> None:
        """Clear the plot."""
        self.plot.clear()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
