from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr
from .common import DISCLAIMER

NonemptyString = Annotated[StrictStr, Field(min_length=1)]


class ClinicalSafetyReport(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    Eligible_For_Vaccine: StrictBool
    Recommended_Dose_mg: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    Reasoning_Steps: Annotated[list[NonemptyString], Field(min_length=1)]
    FDA_Safety_Warnings_Cited: list[NonemptyString]
    Disclaimer: Literal[DISCLAIMER]
