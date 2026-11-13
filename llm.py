# The LLM part: a small open-source model (Qwen2.5, same as in class) turns the
# scouting numbers into a short written report a coach can read before a game.
# It runs on this computer with Hugging Face transformers (no API key needed).
import torch
from transformers import pipeline

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # "Qwen/Qwen2.5-0.5B-Instruct" is faster but writes worse


def load_model(model_id=MODEL_ID):
    # the first run downloads the model (about 3 GB for 1.5B), after that it loads from disk
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # the Mac GPU needs half precision to fit in 8 GB of memory
    dtype = torch.float16 if device == "mps" else torch.float32
    return pipeline("text-generation", model=model_id, dtype=dtype, device=device)


def build_facts(our_name, opp_name, venue, chances, xg, h2h, strong, weak, timing, danger, tips):
    # put every number the report may use into plain text, so the model doesn't invent stats
    lines = [
        f"Match: {our_name} vs {opp_name} ({venue} game for {our_name}).",
        f"Model chances for {our_name}: win {chances['Win']:.0f}%, tie {chances['Tie']:.0f}%, "
        f"loss {chances['Loss']:.0f}%. Expected score {xg[0]:.1f} - {xg[1]:.1f}.",
    ]
    # say the verdict in plain words so the model can't flip it
    favored = our_name if chances["Win"] > chances["Loss"] else opp_name
    lines.insert(1, f"VERDICT: {favored} is favored. {our_name} wins {chances['Win']:.0f}% of the time, "
                    f"loses {chances['Loss']:.0f}%.")
    if h2h.empty:
        lines.append("Past meetings: none in our data.")
    else:
        w, t, l = [(h2h["result"] == r).sum() for r in ["W", "T", "L"]]
        line = f"Past meetings: {our_name} has {w} wins, {t} ties and {l} losses against {opp_name}."
        if w == 0:
            line = line[:-1] + " (never beaten them)."
        lines.append(line)
    lines.append(f"{opp_name} strengths (conference rank): " +
                 "; ".join(f"{label} {value:.1f} (#{rank})" for label, rank, value in strong))
    lines.append(f"{opp_name} weaknesses (conference rank): " +
                 "; ".join(f"{label} {value:.1f} (#{rank})" for label, rank, value in weak))
    if timing is not None and timing.sum().sum() > 0:
        lines.append(f"Goals {opp_name} scored by minute: " +
                     ", ".join(f"{w} {n}" for w, n in timing["Goals scored"].items()))
        lines.append(f"Goals {opp_name} allowed by minute: " +
                     ", ".join(f"{w} {n}" for w, n in timing["Goals allowed"].items()))
    if not danger.empty:
        lines.append(f"{opp_name} key players: " +
                     "; ".join(f"{p} ({r['Goals']} goals, {r['Assists']} assists)" for p, r in danger.iterrows()))
    lines.append("Tactical notes from the stats: " + " ".join(tips))
    return "\n".join(lines)


def scouting_report(generator, facts, temperature=0.2):
    messages = [
        {"role": "system", "content": "You are an assistant coach for a college men's soccer team. "
                                      "You write short, practical scouting reports for players."},
        {"role": "user", "content": f"""Write a scouting report using ONLY the facts below. Do not invent numbers or names. The Overview must agree with the VERDICT line.
Use exactly these three sections with short bullet points:
1. Overview (who is favored and why)
2. Their threats (players and moments of the game to watch)
3. How we win (3 to 4 clear instructions for our players)

Facts:
{facts}"""},
    ]
    # same steps as the class example: chat template -> generate -> keep only the new text
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = generator(prompt, max_new_tokens=400, do_sample=True, temperature=temperature, top_p=0.9)
    return output[0]["generated_text"][len(prompt):].strip()


if __name__ == "__main__":
    # quick test from the terminal: python llm.py
    import time

    import eda
    import scouting as sc

    games, goals = eda.load_games(), eda.load_goals()
    g = sc.pick_games(games, [2025, 2026])
    prof = sc.team_profile(g, 365)
    xg = sc.expected_goals(g, prof, "ncf", "keiser", "home")
    strong, weak, _ = sc.strengths_weaknesses(prof, "keiser")
    timing = sc.goal_timing(goals, "keiser", [2025, 2026])
    facts = build_facts("New College", "Keiser", "home", sc.result_chances(*xg), xg,
                        sc.head_to_head(games, "ncf", "keiser"), strong, weak, timing,
                        sc.dangerous_players(goals, "keiser", [2025, 2026]),
                        sc.game_plan(prof, "ncf", "keiser", timing))
    print(facts, "\n")
    start = time.time()
    generator = load_model()
    print(f"model loaded in {time.time() - start:.0f}s")
    start = time.time()
    print(scouting_report(generator, facts))
    print(f"\nwritten in {time.time() - start:.0f}s")
