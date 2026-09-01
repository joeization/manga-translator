from __future__ import annotations

import tkinter as tk
from queue import Empty, Queue
from threading import Event
from tkinter import ttk

from PIL import Image, ImageTk


class ImageViewer:
    def show(self, image: Image.Image) -> None:
        self.show_pages([image])

    def show_pages(self, images: list[Image.Image]) -> None:
        self.show_stream(Queue(), complete=True, initial_images=images)

    def show_stream(self, images: Queue[Image.Image], cancel_event: Event | None = None, complete: bool = False, initial_images: list[Image.Image] | None = None) -> None:
        root = tk.Tk()
        root.title("Manga Translator")
        root.geometry("900x900")
        canvas = tk.Canvas(root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")
        photos: list[ImageTk.PhotoImage] = []
        pages = initial_images or []

        def render_pages(_: object | None = None) -> None:
            width = max(1, canvas.winfo_width())
            for child in content.winfo_children():
                child.destroy()
            photos.clear()
            for image in pages:
                scale = min(1.0, width / image.width)
                rendered = image if scale == 1 else image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(rendered)
                label = ttk.Label(content, image=photo)
                label.image = photo
                label.pack(fill="x")
                photos.append(photo)
            content.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(canvas_window, width=width)

        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", render_pages)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))
        root.protocol("WM_DELETE_WINDOW", lambda: _close(root, cancel_event))
        def poll() -> None:
            updated = False
            while True:
                try:
                    pages.append(images.get_nowait())
                    updated = True
                except Empty:
                    break
            if updated:
                render_pages()
            if not complete:
                root.after(100, poll)
        poll()
        root.mainloop()


def _close(root: tk.Tk, cancel_event: Event | None) -> None:
    if cancel_event is not None:
        cancel_event.set()
    root.destroy()