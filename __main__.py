"""Local execution entry point. Cloud deployment uses fashion_pose_mcp.py:mcp directly."""

from fashion_pose_mcp import mcp

if __name__ == "__main__":
    mcp.run()
