"""Writer base wrapper class"""

from typing import Any, Optional, Tuple, Union

from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self
from writerai import AsyncWriter, Writer


class BaseWriter(BaseModel):
    client: Writer = Field(default=None, exclude=True)  #: :meta private:
    async_client: AsyncWriter = Field(default=None, exclude=True)  #: :meta private:

    """Writer API key.
    Automatically read from env variable `WRITER_API_KEY` if not provided.
    """
    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "WRITER_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `WRITER_API_KEY`."
            ),
        ),
    )

    """Timeout for requests to Writer completion API. Can be float, httpx.Timeout or
        None."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )

    """Base URL path for API requests."""
    api_base: Optional[str] = Field(
        alias="base_url", default_factory=from_env("WRITER_BASE_URL", default=None)
    )

    """Maximum number of retries to make when generating."""
    max_retries: int = Field(default=2, gt=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists in environment."""

        client_params = {
            "api_key": (self.api_key.get_secret_value() if self.api_key else None),
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "base_url": self.api_base,
        }

        if not self.client:
            self.client = Writer(**client_params)
        if not self.async_client:
            self.async_client = AsyncWriter(**client_params)
        return self
