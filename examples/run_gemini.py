# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# run_gemini.py
import sys
import pathlib
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
from camel.societies import RolePlaying

from owl.utils import run_society, DocumentProcessingToolkit

# Load environment
base_dir = pathlib.Path(__file__).parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))
set_log_level(level="DEBUG")

# FastAPI setup
app = FastAPI(title="Gemini OWL Agent API")

# Optional: Allow cross-origin requests (helpful for frontend devs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str


def construct_society(question: str) -> RolePlaying:
    models = {
        name: ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=ModelType.GEMINI_2_5_PRO_EXP,
            model_config_dict={"temperature": 0},
        )
        for name in [
            "user", "assistant", "browsing", "planning", "video", "image", "document"
        ]
    }

    tools = [
        *BrowserToolkit(headless=True, web_agent_model=models["browsing"], planning_agent_model=models["planning"]).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    return RolePlaying(
        task_prompt=question,
        with_task_specify=False,
        user_role_name="user",
        user_agent_kwargs={"model": models["user"]},
        assistant_role_name="assistant",
        assistant_agent_kwargs={"model": models["assistant"], "tools": tools},
    )


@app.post("/run-task")
async def run_task(request: TaskRequest):
    task = request.task
    society = construct_society(task)
    try:
        answer, chat_history, token_count = run_society(society)
        return JSONResponse({
            "answer": answer,
            "token_count": token_count,
            "chat_history": chat_history,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Gemini OWL API running"}
