import os
from pathlib import Path
from threading import Thread
import tkinter as tk
from tkinter import messagebox,filedialog
from tkinter import ttk
from typing import Any

from src.live_surgery_controller_v1 import (
    LiveSurgeryControllerV1,
)


class LiveSurgeryDemoAppV1:
    """
    Desktop interface for one continuously recorded
    surgical documentation session.

    The interface starts and stops the existing live
    controller. Clinical processing remains inside the
    tested pipeline rather than being reimplemented here.
    """

    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.root.title(
            "OR Documentation Assistant"
        )

        self.root.geometry("1050x760")
        self.root.minsize(900, 650)

        # The controller is created after Start Case is
        # pressed because loading the AI models takes time.
        self.controller: (
            LiveSurgeryControllerV1 | None
        ) = None

        # The lifecycle thread performs model loading,
        # continuous capture, finalization, and cleanup.
        self.case_thread: Thread | None = None

        self.case_active = False
        self.stop_requested = False
        self.window_close_requested = False

        self.final_result: (
            dict[str, Any] | None
        ) = None

        self.report_path: str | None = None

        # Used to display each processed speech segment
        # only once.
        self.displayed_result_count = 0

        self.status_variable = tk.StringVar(
            value="Ready"
        )
        self._configure_style()
        self._build_interface()

        # Poll the controller periodically for new
        # transcripts and extraction results.
        self.root.after(
            300,
            self._poll_controller,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._handle_window_close,
        )

    def _configure_style(self) -> None:
        """
        Configure a simple clinical interface style.
        """

        style = ttk.Style()

        if "vista" in style.theme_names():
            style.theme_use("vista")

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22, "bold"),
            foreground="#164f7c",
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 10),
            foreground="#52606d",
        )

        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 11, "bold"),
            foreground="#176b45",
        )

        style.configure(
            "Metric.TLabel",
            font=("Segoe UI", 10, "bold"),
        )

        style.configure(
            "Start.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(15, 8),
        )

        style.configure(
            "Stop.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(15, 8),
        )

        style.configure(
            "Open.TButton",
            font=("Segoe UI", 10),
            padding=(15, 8),
        )

    def _build_interface(self) -> None:
        """
        Create all visible interface components.
        """

        main_frame = ttk.Frame(
            self.root,
            padding=22,
        )

        main_frame.pack(
            fill="both",
            expand=True,
        )

        ttk.Label(
            main_frame,
            text=(
                "Operating Room Documentation Assistant"
            ),
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            main_frame,
            text=(
                "Create a draft clinical summary from live speech "
                "or an uploaded recording. "
            ),
            style="Subtitle.TLabel",
        ).pack(
            anchor="w",
            pady=(2, 18),
        )

        control_frame = ttk.LabelFrame(
            main_frame,
            text="Case Details and Input",
            padding=14,
        )

        control_frame.pack(
            fill="x",
            pady=(0, 14),
        )

        ttk.Label(
            control_frame,
            text="Case ID:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )

        self.case_id_entry = ttk.Entry(
            control_frame,
            width=32,
            font=("Segoe UI", 11),
        )

        self.case_id_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(0, 12),
        )

        self.case_id_entry.insert(
            0,
            " ",
        )

        self.start_button = ttk.Button(
            control_frame,
            text="Start Live Recording",
            command=self._start_case,
            style="Start.TButton",
        )
        self.upload_button = ttk.Button(
            control_frame,
            text="Upload Recording",
            command=self._upload_recording,
            style="Open.TButton",
        )

        self.upload_button.grid(
            row=0,
            column=3,
            padx=(0, 8),
        )

        self.start_button.grid(
            row=0,
            column=2,
            padx=(0, 8),
        )

        self.end_button = ttk.Button(
            control_frame,
            text="Finish and Create Draft",
            command=self._end_case,
            style="Stop.TButton",
            state="disabled",
        )

        self.end_button.grid(
            row=0,
            column=4,
            padx=(0, 8),
        )

        self.open_report_button = ttk.Button(
            control_frame,
            text="Open Draft Report",
            command=self._open_report,
            style="Open.TButton",
            state="disabled",
        )

        self.open_report_button.grid(
            row=0,
            column=5,
        )

        control_frame.columnconfigure(
            1,
            weight=1,
        )

        status_frame = ttk.Frame(
            main_frame,
        )

        status_frame.pack(
            fill="x",
            pady=(0, 14),
        )

        ttk.Label(
            status_frame,
            text="Status:",
            style="Metric.TLabel",
        ).pack(
            side="left",
        )

        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_variable,
            style="Status.TLabel",
        )

        self.status_label.pack(
            side="left",
            padx=(8, 25),
        )
        content_pane = ttk.Panedwindow(
            main_frame,
            orient="horizontal",
        )

        content_pane.pack(
            fill="both",
            expand=True,
        )

        transcript_frame = ttk.LabelFrame(
            content_pane,
            text="Transcript",
            padding=10,
        )

        extraction_frame = ttk.LabelFrame(
            content_pane,
            text="Validated Clinical Fields",
            padding=10,
        )

        content_pane.add(
            transcript_frame,
            weight=3,
        )

        content_pane.add(
            extraction_frame,
            weight=2,
        )

        self.transcript_text = tk.Text(
            transcript_frame,
            wrap="word",
            font=("Segoe UI", 11),
            background="#ffffff",
            foreground="#202124",
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
        )

        transcript_scrollbar = ttk.Scrollbar(
            transcript_frame,
            orient="vertical",
            command=self.transcript_text.yview,
        )

        self.transcript_text.configure(
            yscrollcommand=(
                transcript_scrollbar.set
            )
        )

        self.transcript_text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        transcript_scrollbar.pack(
            side="right",
            fill="y",
        )

        self.fields_text = tk.Text(
            extraction_frame,
            wrap="word",
            font=("Consolas", 10),
            background="#f5f8fb",
            foreground="#183b56",
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
        )

        fields_scrollbar = ttk.Scrollbar(
            extraction_frame,
            orient="vertical",
            command=self.fields_text.yview,
        )

        self.fields_text.configure(
            yscrollcommand=(
                fields_scrollbar.set
            )
        )

        self.fields_text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        fields_scrollbar.pack(
            side="right",
            fill="y",
        )

        instruction_text = (
            "Start a live recording or upload a saved audio file. "
            "The system will create a draft clinical summary."
        )

        ttk.Label(
            main_frame,
            text=instruction_text,
            style="Subtitle.TLabel",
            wraplength=950,
        ).pack(
            anchor="w",
            pady=(14, 0),
        )
    def _upload_recording(self) -> None:
        """
        Select and process one complete saved recording.
        """

        record_id = self.case_id_entry.get().strip()

        if not record_id:
            messagebox.showerror(
                "Missing Case ID",
                "Enter a case ID before uploading a recording.",
            )
            return

        audio_path = filedialog.askopenfilename(
            title="Select Surgical Recording",
            filetypes=[
                (
                    "Audio files",
                    "*.m4a *.wav *.mp3 *.mp4",
                ),
                ("All files", "*.*"),
            ],
        )

        if not audio_path:
            return

        if self.case_active:
            return

        self._clear_display()

        self.case_active = True
        self.stop_requested = False
        self.final_result = None
        self.report_path = None
        self.controller = None

        self.case_id_entry.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        self.end_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")

        self.status_variable.set(
            "Initializing models and processing uploaded recording..."
        )

        output_directory = (
            Path("outputs")
            / "live_surgery_demo"
        )

        self.case_thread = Thread(
            target=self._run_upload_lifecycle,
            args=(
                record_id,
                output_directory,
                Path(audio_path),
            ),
            name="uploaded-recording-gui-worker",
            daemon=True,
        )

        self.case_thread.start()

    def _start_case(self) -> None:
        """
        Validate the case ID and launch the complete live
        workflow in a background thread.
        """

        record_id = (
            self.case_id_entry.get().strip()
        )

        if not record_id:
            messagebox.showerror(
                "Missing Case ID",
                "Enter a case ID before starting.",
            )

            return

        if self.case_active:
            return

        self._clear_display()

        self.case_active = True
        self.stop_requested = False
        self.final_result = None
        self.report_path = None
        self.controller = None

        self.case_id_entry.configure(
            state="disabled"
        )

        self.start_button.configure(
            state="disabled"
        )

        self.end_button.configure(
            state="disabled"
        )

        self.open_report_button.configure(
            state="disabled"
        )

        self.status_variable.set(
            "Initializing Whisper, BioLORD, and VAD..."
        )

        output_directory = (
            Path("outputs")
            / "live_surgery_demo"
        )

        self.case_thread = Thread(
            target=self._run_case_lifecycle,
            args=(
                record_id,
                output_directory,
            ),
            name="live-surgery-gui-worker",
            daemon=True,
        )

        self.case_thread.start()
    def _run_upload_lifecycle(
        self,
        record_id: str,
        output_directory: Path,
        audio_path: Path,
    ) -> None:
        """
        Process one uploaded recording in a background thread.
        """

        try:
            controller = LiveSurgeryControllerV1(
                record_id=record_id,
                output_directory=output_directory,
                input_device=None,
                whisper_model_name="turbo",
                language="en",
                top_k=3,
                silence_duration_ms=1000,
                speech_threshold=0.5,
            )

            self.controller = controller

            self.root.after(
                0,
                lambda: self.status_variable.set(
                    "Transcribing and extracting clinical fields..."
                ),
            )

            controller.process_uploaded_recording(
                audio_path=audio_path,
            )

            self.root.after(
                0,
                lambda: self.status_variable.set(
                    "Finalizing case and generating report..."
                ),
            )

            final_result = controller.finalize_case()

            self.final_result = final_result
            self.report_path = final_result.get(
                "report_output_path"
            )

            self.root.after(
                0,
                self._mark_completed,
            )

        except Exception as error:
            error_message = (
                f"{type(error).__name__}: {error}"
            )

            self.root.after(
                0,
                lambda message=error_message: (
                    self._show_case_error(message)
                ),
            )

        finally:
            if self.controller is not None:
                self.controller.close()

    def _run_case_lifecycle(
        self,
        record_id: str,
        output_directory: Path,
    ) -> None:
        """
        Run the blocking clinical controller outside the
        Tkinter main thread.

        Tkinter widgets are never changed directly here.
        root.after schedules interface changes safely on
        the main GUI thread.
        """

        try:
            controller = LiveSurgeryControllerV1(
                record_id=record_id,
                output_directory=output_directory,
                input_device=None,
                whisper_model_name="turbo",
                language="en",
                top_k=3,
                silence_duration_ms=1000,
                speech_threshold=0.5,
            )

            self.controller = controller

            self.root.after(
                0,
                self._mark_listening,
            )

            controller.run_live_capture(
                maximum_listening_seconds=None,
            )

            self.root.after(
                0,
                lambda: self.status_variable.set(
                    "Finalizing case and generating report..."
                ),
            )

            final_result = (
                controller.finalize_case()
            )

            self.final_result = final_result
            self.report_path = final_result.get(
                "report_output_path"
            )

            self.root.after(
                0,
                self._mark_completed,
            )

        except Exception as error:
            error_message = (
                f"{type(error).__name__}: {error}"
            )

            self.root.after(
                0,
                lambda message=error_message: (
                    self._show_case_error(message)
                ),
            )

        finally:
            if self.controller is not None:
                self.controller.close()

    def _mark_listening(self) -> None:
        """
        Update controls after model initialization.
        """

        self.status_variable.set(
            "Listening continuously"
        )

        self.end_button.configure(
            state="normal"
        )

    def _end_case(self) -> None:
        """
        Ask the recorder to stop.

        The controller then waits for queued segments,
        finalizes Schema V2, saves MongoDB and JSON, and
        generates the HTML report.
        """

        if (
            not self.case_active
            or self.controller is None
        ):
            return

        if self.stop_requested:
            return

        self.stop_requested = True

        self.status_variable.set(
            "Stopping capture and processing remaining audio..."
        )

        self.end_button.configure(
            state="disabled"
        )

        self.controller.request_stop()

    def _poll_controller(self) -> None:
        """
        Periodically copy newly processed results into
        the interface.
        """

        controller = self.controller

        if controller is not None:
            results_snapshot = list(
                controller.processing_results
            )

            while (
                self.displayed_result_count
                < len(results_snapshot)
            ):
                result = results_snapshot[
                    self.displayed_result_count
                ]

                self._display_processing_result(
                    result
                )

                self.displayed_result_count += 1
        if self.window_close_requested:
            thread_finished = (
                self.case_thread is None
                or not self.case_thread.is_alive()
            )

            if thread_finished:
                self.root.destroy()
                return

        self.root.after(
            300,
            self._poll_controller,
        )

    def _display_processing_result(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Display one processed speech segment and all of
        its validated extractions.
        """

        transcript = str(
            result.get(
                "transcript",
                "",
            )
        ).strip()

        if transcript:
            transcript_block = (
                f"{transcript}\n\n"
            )

            self._append_text(
                widget=self.transcript_text,
                text=transcript_block,
            )

        accepted_extractions = result.get(
            "accepted_extractions",
            [],
        )

        for extraction in accepted_extractions:
            field_name = extraction.get(
                "field",
                "unknown_field",
            )

            value = extraction.get(
                "value"
            )

            readable_field_name = (
                field_name.replace("_", " ").title()
            )

            displayed_value = self._format_display_value(
                value
            )

            field_block = (
                f"{readable_field_name}: "
                f"{displayed_value}\n\n"
            )

            self._append_text(
                widget=self.fields_text,
                text=field_block,
            )

    def _append_text(
        self,
        widget: tk.Text,
        text: str,
    ) -> None:
        """
        Append text to a read-only text box.
        """

        widget.configure(
            state="normal"
        )

        widget.insert(
            "end",
            text,
        )

        widget.see("end")

        widget.configure(
            state="disabled"
        )

    def _mark_completed(self) -> None:
        """
        Update the interface after successful report
        generation.
        """

        self.case_active = False

        self.status_variable.set(
            "Completed — report generated"
        )

        self.case_id_entry.configure(
            state="normal"
        )
        self.upload_button.configure(
            state="normal"
        )

        self.start_button.configure(
            state="normal"
        )

        self.end_button.configure(
            state="disabled"
        )

        if self.report_path:
            self.open_report_button.configure(
                state="normal"
            )

        if not self.window_close_requested:
            messagebox.showinfo(
                "Case Completed",
                (
                    "The structured surgical record "
                    "and clinical report were generated "
                    "successfully."
                ),
            )

    def _show_case_error(
        self,
        error_message: str,
    ) -> None:
        """
        Restore the interface after a workflow failure.
        """

        self.case_active = False

        self.status_variable.set(
            "Error"
        )

        self.case_id_entry.configure(
            state="normal"
        )

        self.start_button.configure(
            state="normal"
        )

        self.end_button.configure(
            state="disabled"
        )

        if not self.window_close_requested:
            messagebox.showerror(
                "Live Case Error",
                error_message,
            )

    def _open_report(self) -> None:
        """
        Open the generated HTML report using the default
        Windows web browser.
        """

        if not self.report_path:
            return

        report_path = Path(
            self.report_path
        ).resolve()

        if not report_path.exists():
            messagebox.showerror(
                "Report Not Found",
                f"Report does not exist:\n{report_path}",
            )

            return

        os.startfile(
            str(report_path)
        )

    def _clear_display(self) -> None:
        """
        Reset visible results before starting a new case.
        """

        self.displayed_result_count = 0
        for widget in (
            self.transcript_text,
            self.fields_text,
        ):
            widget.configure(
                state="normal"
            )

            widget.delete(
                "1.0",
                "end",
            )

            widget.configure(
                state="disabled"
            )

    def _handle_window_close(self) -> None:
        """
        Stop an active recording safely before closing.
        """

        if not self.case_active:
            self.root.destroy()
            return

        should_close = messagebox.askyesno(
            "Close Active Case",
            (
                "A surgical case is still active. "
                "Stop recording, finalize the case, "
                "and close the application?"
            ),
        )

        if not should_close:
            return

        self.window_close_requested = True
        self.status_variable.set(
            "Stopping active case before closing..."
        )

        if (
            self.controller is not None
            and not self.stop_requested
        ):
            self.stop_requested = True
            self.end_button.configure(
                state="disabled"
            )

            self.controller.request_stop()
    def _format_display_value(
        self,
        value: Any,
    ) -> str:
        if value is True:
            return "Yes"

        if value is False:
            return "No"

        if isinstance(value, list):
            return ", ".join(
                str(item)
                for item in value
            )

        return str(value)


def main() -> None:
    """
    Start the desktop application.
    """

    root = tk.Tk()

    LiveSurgeryDemoAppV1(
        root=root
    )

    root.mainloop()


if __name__ == "__main__":
    main()