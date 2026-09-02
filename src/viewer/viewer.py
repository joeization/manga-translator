"""
Tkinter-based ImageViewer for continuous downward scrolling of translated manga pages.
Uses virtualized viewport rendering to eliminate Windows GDI 32,767px canvas height limitations.
"""
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

    def show_stream(
        self,
        images: Queue[Image.Image],
        cancel_event: Event | None = None,
        complete: bool = False,
        initial_images: list[Image.Image] | None = None,
    ) -> None:
        root = tk.Tk()
        root.title("Manga Translator Viewer")
        root.geometry("980x980")

        canvas = tk.Canvas(root, highlightthickness=0, bg="#1e1e1e")
        scrollbar = ttk.Scrollbar(root, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        pages: list[Image.Image] = list(initial_images or [])
        page_scaled_sizes: list[tuple[int, int]] = []
        page_y_offsets: list[int] = []
        total_content_height = 0

        visible_photos: dict[int, ImageTk.PhotoImage] = {}
        visible_canvas_items: dict[int, int] = {}

        scroll_y = 0
        viewport_width = 1
        viewport_height = 1
        rendered_width = 0
        resize_timer: str | None = None

        def recalculate_layout() -> None:
            nonlocal page_scaled_sizes, page_y_offsets, total_content_height, viewport_width, viewport_height, rendered_width
            viewport_width = max(1, canvas.winfo_width())
            viewport_height = max(1, canvas.winfo_height())

            page_scaled_sizes = []
            page_y_offsets = []
            current_y = 0

            for img in pages:
                scale = min(1.0, viewport_width / img.width)
                sw = max(1, round(img.width * scale))
                sh = max(1, round(img.height * scale))
                page_scaled_sizes.append((sw, sh))
                page_y_offsets.append(current_y)
                current_y += sh + 10  # 10px spacing between pages

            total_content_height = current_y
            rendered_width = viewport_width
            update_scrollbar()

        def update_scrollbar() -> None:
            if total_content_height <= viewport_height:
                scrollbar.set(0.0, 1.0)
            else:
                top = scroll_y / total_content_height
                bottom = (scroll_y + viewport_height) / total_content_height
                scrollbar.set(top, min(1.0, bottom))

        def render_viewport() -> None:
            nonlocal scroll_y
            if not pages:
                canvas.delete("all")
                visible_photos.clear()
                visible_canvas_items.clear()
                return

            if viewport_width != rendered_width or len(page_scaled_sizes) != len(pages):
                recalculate_layout()

            max_scroll = max(0, total_content_height - viewport_height)
            scroll_y = max(0, min(scroll_y, max_scroll))
            update_scrollbar()

            v_top = scroll_y - viewport_height
            v_bottom = scroll_y + viewport_height * 2

            currently_visible: set[int] = set()
            for idx, y_top in enumerate(page_y_offsets):
                sw, sh = page_scaled_sizes[idx]
                y_bot = y_top + sh
                if y_bot >= v_top and y_top <= v_bottom:
                    currently_visible.add(idx)

            to_remove = set(visible_canvas_items.keys()) - currently_visible
            for idx in to_remove:
                canvas.delete(visible_canvas_items[idx])
                del visible_canvas_items[idx]
                del visible_photos[idx]

            for idx in currently_visible:
                sw, sh = page_scaled_sizes[idx]
                y_top = page_y_offsets[idx]
                item_canvas_y = y_top - scroll_y
                item_canvas_x = max(0, (viewport_width - sw) // 2)

                if idx in visible_canvas_items:
                    canvas.coords(visible_canvas_items[idx], item_canvas_x, item_canvas_y)
                else:
                    img = pages[idx]
                    if sw == img.width and sh == img.height:
                        rendered = img
                    else:
                        rendered = img.resize((sw, sh), Image.Resampling.BILINEAR)

                    photo = ImageTk.PhotoImage(rendered)
                    item_id = canvas.create_image(item_canvas_x, item_canvas_y, anchor="nw", image=photo)
                    visible_photos[idx] = photo
                    visible_canvas_items[idx] = item_id

        def scroll_by(delta_pixels: int) -> None:
            nonlocal scroll_y
            max_scroll = max(0, total_content_height - viewport_height)
            new_y = max(0, min(scroll_y + delta_pixels, max_scroll))
            if new_y != scroll_y:
                scroll_y = new_y
                render_viewport()

        def on_scrollbar_action(*args: str) -> None:
            nonlocal scroll_y
            max_scroll = max(0, total_content_height - viewport_height)
            if not args:
                return
            action = args[0]
            if action == "moveto":
                fraction = float(args[1])
                scroll_y = round(fraction * total_content_height)
            elif action == "scroll":
                number = int(args[1])
                unit = args[2]
                if unit == "units":
                    scroll_by(number * 60)
                elif unit == "pages":
                    scroll_by(number * viewport_height)
            render_viewport()

        scrollbar.config(command=on_scrollbar_action)

        def on_mouse_wheel(event: tk.Event) -> None:
            delta = event.delta
            if delta != 0:
                scroll_by(-int(delta / 120) * 80)

        def on_canvas_configure(_: tk.Event | None = None) -> None:
            nonlocal resize_timer
            if resize_timer is not None:
                try:
                    root.after_cancel(resize_timer)
                except Exception:
                    pass
            resize_timer = root.after(100, render_viewport)

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        canvas.bind_all("<Button-4>", lambda _: scroll_by(-80))
        canvas.bind_all("<Button-5>", lambda _: scroll_by(80))

        root.bind("<Up>", lambda _: scroll_by(-80))
        root.bind("<Down>", lambda _: scroll_by(80))
        root.bind("<Prior>", lambda _: scroll_by(-viewport_height))  # PageUp
        root.bind("<Next>", lambda _: scroll_by(viewport_height))    # PageDown
        root.bind("<space>", lambda _: scroll_by(viewport_height))
        root.bind("<Home>", lambda _: scroll_by(-total_content_height))
        root.bind("<End>", lambda _: scroll_by(total_content_height))

        root.protocol("WM_DELETE_WINDOW", lambda: _close(root, cancel_event))

        def poll() -> None:
            updated = False
            while True:
                try:
                    image = images.get_nowait()
                    pages.append(image)
                    updated = True
                except Empty:
                    break

            if updated:
                recalculate_layout()
                render_viewport()

            if not complete:
                root.after(100, poll)

        render_viewport()
        poll()
        root.mainloop()


def _close(root: tk.Tk, cancel_event: Event | None) -> None:
    if cancel_event is not None:
        cancel_event.set()
    root.destroy()