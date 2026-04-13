"""
exception.py
────────────
Custom exception for the Cancer Risk pipeline.
Captures file name and line number automatically.
"""
import sys


class CancerRiskException(Exception):
    def __init__(self, error_message: str, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = self._format(error_message, error_detail)

    @staticmethod
    def _format(message: str, error_detail: sys) -> str:
        _, _, tb = error_detail.exc_info()
        if tb is not None:
            file_name = tb.tb_frame.f_code.co_filename
            line_no   = tb.tb_lineno
            return (
                f"Error in [{file_name}] "
                f"at line [{line_no}]: {message}"
            )
        return message

    def __str__(self):
        return self.error_message
