"""MCP stdio server for ADMM Lasso Solver."""
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from client import ADMMLasso

def handle_rpc(request):
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "solve_admm_lasso",
                        "description": "Solve L1-regularized Lasso regression via ADMM",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "b": {"type": "array", "items": {"type": "number"}},
                                "lam": {"type": "number", "default": 0.1},
                                "rho": {"type": "number", "default": 1.0}
                            },
                            "required": ["A", "b"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "solve_admm_lasso":
            A = args.get("A", [])
            b = args.get("b", [])
            lam = float(args.get("lam", 0.1))
            rho = float(args.get("rho", 1.0))
            res = ADMMLasso.solve(A, b, lam, rho)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {name} not found"}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_rpc(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
