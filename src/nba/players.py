from __future__ import annotations

from functools import lru_cache

from nba_api.stats.static import players


@lru_cache(maxsize=1)
def get_all_players(active_only: bool = False) -> list[dict]:
    """Load players from nba_api static metadata."""
    all_players = players.get_players()
    if active_only:
        return [player for player in all_players if player.get("is_active")]
    return all_players


@lru_cache(maxsize=1)
def get_player_name_to_id() -> dict[str, int]:
    """Create a name -> id lookup for UI selection."""
    return {player["full_name"]: int(player["id"]) for player in get_all_players(active_only=False)}


def get_player_names(active_only: bool = False) -> list[str]:
    """Return sorted list of player names for selectbox autocomplete."""
    return sorted(player["full_name"] for player in get_all_players(active_only=active_only))


def resolve_player_id(player_name: str) -> int:
    """Resolve player id from a user-entered player name."""
    if not player_name:
        raise ValueError("Player name cannot be empty.")

    name_to_id = get_player_name_to_id()
    if player_name in name_to_id:
        return name_to_id[player_name]

    lower = player_name.lower().strip()
    exact_casefold = {name.lower(): pid for name, pid in name_to_id.items()}
    if lower in exact_casefold:
        return exact_casefold[lower]

    contains_matches = [
        (name, pid) for name, pid in name_to_id.items() if lower in name.lower()
    ]
    if len(contains_matches) == 1:
        return int(contains_matches[0][1])
    if len(contains_matches) > 1:
        suggestions = ", ".join(name for name, _ in contains_matches[:5])
        raise ValueError(
            f"Player name is ambiguous. Try one of: {suggestions}"
        )

    raise ValueError(f"Player '{player_name}' was not found in nba_api static player list.")
