#!/usr/bin/env python3
"""
PreProConvert Server
Run with: python run_server.py
"""

import argparse
import uvicorn
import webbrowser
from threading import Timer


def open_browser(url: str):
    """Open browser after server starts"""
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="PreProConvert Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      PreProConvert                          ║
╠══════════════════════════════════════════════════════════════╣
║  Server starting at: {url:<38} ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Open browser after short delay (gives server time to start)
    if not args.no_browser:
        Timer(1.5, open_browser, [url]).start()

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
