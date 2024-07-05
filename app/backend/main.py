from langchain_openai import AzureChatOpenAI, ChatOpenAI
import warnings
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import create_agent

warnings.filterwarnings('ignore')

## Use this version if you want to use AzureChatOpenAI
api_version = "2024-02-15-preview"
DEPLOYMENT_NAME = "gpt-35-turbo-default"
llm = AzureChatOpenAI(
    openai_api_version=api_version,
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_type="azure",
    streaming=True,
    temperature=0.0
)


# ## Use this to run the program locally
# llm = ChatOpenAI(
#     streaming=True,
#     temperature=0.0
# )

agent = create_agent(llm=llm)

app = FastAPI()

# Add CORS
origin = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


class Message(BaseModel):
    content: str


async def generate(query: str):
    async for token in agent.astream(input={"input": query}):
        if "output" in token.keys():
            yield token["output"]


@app.post("/query/")
async def get_response(query: Message = ...):
    gen = generate(query.content)
    return StreamingResponse(gen, media_type="text/event-stream")


@app.get("/health")
async def get_health():
    return {"Still here :)"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = app.port
    print("The port used for this app is", port)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
