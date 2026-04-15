from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
import json
import os
import math
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─── Load Graph Data ──────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "graph_data.json")) as f:
    GRAPH_DATA = json.load(f)

# ─── Build NetworkX Graph ─────────────────────────────────────────────────────

def build_graph(time_of_day: str = "day"):
    """
    Build a directed graph from road network data.
    Edge weights are computed differently for 'fastest' vs 'safest' routing.
    """
    G = nx.DiGraph()

    for node in GRAPH_DATA["nodes"]:
        G.add_node(node["id"], label=node["label"], lat=node["lat"], lng=node["lng"])

    for edge in GRAPH_DATA["edges"]:
        risk_score = compute_risk(edge, time_of_day)
        G.add_edge(
            edge["from"], edge["to"],
            travel_time=edge["travel_time"],
            distance=edge["distance"],
            lighting_score=edge["lighting_score"],
            crowd_density=edge["crowd_density"],
            incident_history=edge["incident_history"],
            congestion_level=edge["congestion_level"],
            road_name=edge["road_name"],
            risk_score=risk_score
        )
        # Make bidirectional (undirected road behavior)
        G.add_edge(
            edge["to"], edge["from"],
            travel_time=edge["travel_time"],
            distance=edge["distance"],
            lighting_score=edge["lighting_score"],
            crowd_density=edge["crowd_density"],
            incident_history=edge["incident_history"],
            congestion_level=edge["congestion_level"],
            road_name=edge["road_name"],
            risk_score=risk_score
        )

    return G


def compute_risk(edge: dict, time_of_day: str) -> float:
    """
    Risk score: lower = safer.
    Components:
      - lighting_score (1–10): higher = safer → invert
      - crowd_density (1–10): moderate is safe; extremes are risky
      - incident_history (count): higher = riskier
      - congestion_level (1–10): higher = more risky (accidents)

    At night: lighting and incident_history weighted more heavily.
    """
    lighting   = edge["lighting_score"]      # 1–10, higher = better
    crowd      = edge["crowd_density"]        # 1–10
    incidents  = edge["incident_history"]     # raw count
    congestion = edge["congestion_level"]     # 1–10

    if time_of_day == "night":
        # Night: lighting & incidents dominate
        w_lighting   = 0.40
        w_crowd      = 0.10
        w_incidents  = 0.35
        w_congestion = 0.15
    else:
        # Day: balanced
        w_lighting   = 0.20
        w_crowd      = 0.25
        w_incidents  = 0.30
        w_congestion = 0.25

    # Normalize incident_history to 0–10 scale (cap at 10)
    incidents_norm = min(incidents, 10)

    # Lighting is inverted: poor lighting → high risk
    lighting_risk = (10 - lighting)

    # Crowd risk: empty streets at night risky, very crowded also risky
    if time_of_day == "night":
        crowd_risk = 10 - crowd  # less crowd at night = higher risk
    else:
        crowd_risk = abs(crowd - 5) * 2  # moderate crowd is safest

    risk = (
        w_lighting   * lighting_risk +
        w_crowd      * crowd_risk +
        w_incidents  * incidents_norm +
        w_congestion * congestion
    )
    return round(risk, 3)


def get_route_details(G, path: list) -> dict:
    """Aggregate stats for a given node path."""
    segments = []
    total_time = 0
    total_risk = 0
    total_distance = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        segments.append({
            "from": G.nodes[u]["label"],
            "to": G.nodes[v]["label"],
            "road_name": data["road_name"],
            "travel_time": data["travel_time"],
            "risk_score": data["risk_score"],
            "lighting_score": data["lighting_score"],
            "incident_history": data["incident_history"],
            "congestion_level": data["congestion_level"]
        })
        total_time += data["travel_time"]
        total_risk += data["risk_score"]
        total_distance += data["distance"]

    avg_risk = round(total_risk / len(segments), 2) if segments else 0

    return {
        "nodes": [G.nodes[n]["label"] for n in path],
        "segments": segments,
        "total_travel_time": total_time,
        "total_risk_score": round(total_risk, 2),
        "avg_segment_risk": avg_risk,
        "total_distance_km": round(total_distance, 2)
    }


def generate_ai_explanation(fastest: dict, safest: dict, time_of_day: str) -> str:
    """Generate a deterministic, human-readable explanation of route choice."""
    risk_diff = round(fastest["total_risk_score"] - safest["total_risk_score"], 2)
    time_diff = safest["total_travel_time"] - fastest["total_travel_time"]

    # Identify the riskiest segment on fastest route
    if fastest["segments"]:
        worst_seg = max(fastest["segments"], key=lambda s: s["risk_score"])
        worst_road = worst_seg["road_name"]
        worst_risk = worst_seg["risk_score"]
    else:
        worst_road = "unknown"
        worst_risk = 0

    # Best segment on safest route
    if safest["segments"]:
        best_seg = min(safest["segments"], key=lambda s: s["risk_score"])
        best_road = best_seg["road_name"]
    else:
        best_road = "unknown"

    time_str = f"{time_diff} min longer" if time_diff > 0 else "about the same time"
    night_note = (
        " At night, poor lighting and high incident history on the faster path pose elevated risks."
        if time_of_day == "night" else ""
    )

    explanation = (
        f"The safest route avoids {worst_road} (risk score: {worst_risk:.1f}), "
        f"which has {'poor lighting and ' if worst_seg.get('lighting_score', 10) < 6 else ''}"
        f"high incident history ({worst_seg.get('incident_history', 0)} incidents). "
        f"Instead, it routes via {best_road} with significantly lower risk. "
        f"The safety-optimised path is {time_str} but reduces overall risk by "
        f"{risk_diff:.1f} points ({round(risk_diff / max(fastest['total_risk_score'], 0.01) * 100, 1)}%).{night_note}"
    )
    return explanation


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "UrbanShield AI backend running", "version": "1.0.0"})


@app.route("/nodes", methods=["GET"])
def get_nodes():
    return jsonify(GRAPH_DATA["nodes"])


@app.route("/recommend-route", methods=["POST"])
def recommend_route():
    body = request.get_json()
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    source = body.get("source")
    target = body.get("target")
    time_of_day = body.get("time_of_day", "auto")

    # Auto-detect time of day
    if time_of_day == "auto":
        hour = datetime.now().hour
        time_of_day = "night" if (hour < 6 or hour >= 20) else "day"

    if time_of_day not in ("day", "night"):
        return jsonify({"error": "time_of_day must be 'day', 'night', or 'auto'"}), 400

    valid_nodes = {n["id"] for n in GRAPH_DATA["nodes"]}
    if source not in valid_nodes:
        return jsonify({"error": f"Unknown source node: {source}"}), 400
    if target not in valid_nodes:
        return jsonify({"error": f"Unknown target node: {target}"}), 400
    if source == target:
        return jsonify({"error": "Source and target must be different"}), 400

    G = build_graph(time_of_day)

    # ── Fastest Route: minimise travel_time ──
    try:
        fastest_path = nx.dijkstra_path(G, source, target, weight="travel_time")
        fastest = get_route_details(G, fastest_path)
        fastest["route_type"] = "fastest"
    except nx.NetworkXNoPath:
        return jsonify({"error": f"No path exists from {source} to {target}"}), 404

    # ── Safest Route: minimise risk_score ──
    try:
        safest_path = nx.dijkstra_path(G, source, target, weight="risk_score")
        safest = get_route_details(G, safest_path)
        safest["route_type"] = "safest"
    except nx.NetworkXNoPath:
        safest = fastest.copy()
        safest["route_type"] = "safest"

    # ── AI Explanation ──
    explanation = generate_ai_explanation(fastest, safest, time_of_day)

    are_same = fastest_path == safest_path

    return jsonify({
        "source": source,
        "target": target,
        "time_of_day": time_of_day,
        "routes_are_identical": are_same,
        "fastest_route": fastest,
        "safest_route": safest,
        "ai_explanation": explanation,
        "meta": {
            "nodes_in_graph": G.number_of_nodes(),
            "edges_in_graph": G.number_of_edges() // 2,
            "algorithm": "Dijkstra (NetworkX)"
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)