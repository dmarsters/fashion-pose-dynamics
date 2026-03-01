"""
Fashion Pose Dynamics MCP Server

Canonical fashion poses as named geometric configurations with
deterministic parameter mappings in 7D pose-space. Computes
compositional geometry, lighting interaction, and exports body
surface maps for cross-domain composition.

Three-layer olog architecture:
- Layer 1: Pure taxonomy lookup (0 tokens)
- Layer 2: Deterministic computation (0 tokens)
- Layer 3: Structured data for Claude synthesis (~100-200 tokens)
"""

from fastmcp import FastMCP

mcp = FastMCP("fashion-pose-dynamics")

from layers.taxonomy import register_taxonomy_tools
from layers.computation import register_computation_tools
from layers.synthesis import register_synthesis_tools

register_taxonomy_tools(mcp)
register_computation_tools(mcp)
register_synthesis_tools(mcp)

if __name__ == "__main__":
    mcp.run()
