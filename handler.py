import asyncio
from fastapi import FastAPI, HTTPException
from app import load_models, TryOnRequest, process_images_standalone

class ServerlessHandler:
    def __init__(self):
        self.app = FastAPI(title="Virtual Try-On API")
        self.pipe = None
        self.ready = False
        self._setup()

    def _setup(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "ready" if self.ready else "loading"}

        @self.app.post("/try-on/")
        async def try_on(request: TryOnRequest):
            if not self.ready:
                raise HTTPException(503, "Service initializing")
            return process_images_standalone(self.pipe, **request.dict())

        @self.app.get("/")
        async def root():
            return {"message": "Virtual Try-On API"}

    async def initialize(self):
        """Parallel initialization"""
        server_task = asyncio.create_task(self._start_server())
        model_task = asyncio.create_task(self._load_models())
        await asyncio.gather(server_task, model_task)
        self.ready = True

    async def _load_models(self):
        """Async model loading with retries"""
        try:
            self.pipe = load_models()
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise

    async def _start_server(self):
        """Non-blocking server startup"""
        from uvicorn import Config, Server
        config = Config(
            app=self.app,
            host="0.0.0.0",
            port=8000,
            timeout_keep_alive=600,
            limit_max_requests=1000
        )
        server = Server(config)
        await server.serve()

async def main():
    handler = ServerlessHandler()
    await handler.initialize()

if __name__ == "__main__":
    asyncio.run(main())

