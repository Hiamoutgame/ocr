"""
Post-processing utilities for transforming raw OCR text into structured data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from core.interfaces import DataParser


@dataclass(slots=True)
class FinancialPeriodData:
    """Container for data of a single financial period (start or end of year)."""

    date: Optional[str]
    data: Dict[str, float]


@dataclass(slots=True)
class FinancialReport:
    """
    High-level representation of a financial report to be persisted to NoSQL.
    """

    report_id: str
    start_of_period: FinancialPeriodData
    end_of_period: FinancialPeriodData
    calculated_metrics: Dict[str, float]


class FinancialReportParser(DataParser):
    """
    Parse raw OCR text of a Vietnamese financial report into structured JSON-like data.

    Expected output shape (example):

    {
        "report_id": "uuid...",
        "financial_period": {
            "start_of_period": {
                "date": "01/01/202X",
                "data": {
                    "tong_tai_san": 1000000000,
                    "no_phai_tra": 500000000
                }
            },
            "end_of_period": {
                "date": "31/12/202X",
                "data": {
                    "tong_tai_san": 1200000000,
                    "no_phai_tra": 600000000
                }
            }
        },
        "calculated_metrics": {
            "growth_rate": 0.2
        }
    }
    """

    def parse(self, text: str) -> dict[str, Any]:
        """
        Entry point for transforming raw OCR text into a JSON-serializable dict.

        NOTE: This is a skeleton implementation; you can gradually fill in
        the regex and rule-based extraction logic in the private helpers below.
        """
        report_id = str(uuid4())

        start_period = self._extract_start_of_period(text)
        end_period = self._extract_end_of_period(text)
        metrics = self._calculate_metrics(start_period, end_period)

        return {
            "report_id": report_id,
            "financial_period": {
                "start_of_period": {
                    "date": start_period.date,
                    "data": start_period.data,
                },
                "end_of_period": {
                    "date": end_period.date,
                    "data": end_period.data,
                },
            },
            "calculated_metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Private helpers (to be implemented with regex / NLP rules)
    # ------------------------------------------------------------------

    def _extract_start_of_period(self, text: str) -> FinancialPeriodData:
        """
        Extract numbers corresponding to 'Số đầu năm' and related fields.

        TODO:
            - Use regex to locate the 'Số đầu năm' column.
            - Map Vietnamese row labels (e.g. 'Tổng tài sản', 'Nợ phải trả')
              to normalized snake_case keys (e.g. 'tong_tai_san').
            - Parse numeric values, normalizing thousand separators and
              decimal commas.
        """
        # Placeholder implementation; returns empty structure.
        return FinancialPeriodData(date=None, data={})

    def _extract_end_of_period(self, text: str) -> FinancialPeriodData:
        """
        Extract numbers corresponding to 'Số cuối năm' and related fields.

        TODO:
            - Similar to _extract_start_of_period, but targeting the
              'Số cuối năm' column.
        """
        # Placeholder implementation; returns empty structure.
        return FinancialPeriodData(date=None, data={})

    def _calculate_metrics(
        self,
        start_period: FinancialPeriodData,
        end_period: FinancialPeriodData,
    ) -> Dict[str, float]:
        """
        Derive metrics such as growth_rate from the extracted data.

        Example:
            growth_rate = (end_total_assets - start_total_assets) / start_total_assets
        """
        metrics: Dict[str, float] = {}

        # Example skeleton for future implementation:
        # start_assets = start_period.data.get("tong_tai_san")
        # end_assets = end_period.data.get("tong_tai_san")
        # if start_assets and end_assets and start_assets != 0:
        #     metrics["growth_rate"] = (end_assets - start_assets) / start_assets

        return metrics

