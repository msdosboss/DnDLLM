import json
import random
import argparse
from copy import deepcopy
from collections import deque
from PIL import Image, ImageDraw, ImageFont

TERRAINS = ["grass", "stone", "water", "wood"]
TRAPS = ["none", "spike trap", "hidden spike trap"]

def make_empty_map(width, height):
    return [[{"terrain": "grass", "trap": "none", "height": 0, "creatures": "none"} for _ in range(width)] for _ in range(height)]

def spread_terrain(map_data, terrain, attempts, spread_chance=0.6, min_distance=0):
    height = len(map_data)
    width = len(map_data[0])

    for _ in range(attempts):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Tree spacing check
        if terrain == "wood" and min_distance > 0:
            too_close = any(
                map_data[yy][xx]["terrain"] == "wood"
                for yy in range(max(0, y - min_distance), min(height, y + min_distance + 1))
                for xx in range(max(0, x - min_distance), min(width, x + min_distance + 1))
                if (xx, yy) != (x, y)
            )
            if too_close:
                continue

        map_data[y][x]["terrain"] = terrain

        # Spread from the point
        frontier = [(x, y)]
        for _ in range(10):  # limit spread depth
            new_frontier = []
            for fx, fy in frontier:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx + dx, fy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if map_data[ny][nx]["terrain"] == "grass" and random.random() < spread_chance:
                            map_data[ny][nx]["terrain"] = terrain
                            new_frontier.append((nx, ny))
            frontier = new_frontier

def load_generation_settings(filename="gen-settings.json"):
    with open(filename, "r") as f:
        return json.load(f)

def generate_height_layer(grid, height_settings):
    height = len(grid)
    width = len(grid[0])

    seed_min = height_settings.get("seed_height_min", 6)
    seed_max = height_settings.get("seed_height_max", 10)
    norm_max = height_settings.get("normalization_max", 10)

    # 1. Seed heights
    height_map = [[None for _ in range(width)] for _ in range(height)]
    num_seeds = 3
    for _ in range(num_seeds):
        sx, sy = random.randint(0, width - 1), random.randint(0, height - 1)
        seed_height = random.randint(seed_min, seed_max)
        height_map[sy][sx] = seed_height

    def neighbors(x, y):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    # 2. Propagate heights
    queue = deque()
    for y in range(height):
        for x in range(width):
            if height_map[y][x] is not None:
                queue.append((x, y, height_map[y][x]))

    while queue:
        x, y, h = queue.popleft()
        for nx, ny in neighbors(x, y):
            if height_map[ny][nx] is None or height_map[ny][nx] > h + 1:
                height_map[ny][nx] = h + 1
                queue.append((nx, ny, h + 1))

    # 3. Normalize
    max_h = max(h for row in height_map for h in row if h is not None)
    for y in range(height):
        for x in range(width):
            height_map[y][x] = int(norm_max * height_map[y][x] / max_h)

    # 4. Water is always 0 and nearby land tiles are smoothed
    for y in range(height):
        for x in range(width):
            if grid[y][x]["terrain"] == "water":
                height_map[y][x] = 0

    for y in range(height):
        for x in range(width):
            if grid[y][x]["terrain"] != "water":
                for nx, ny in neighbors(x, y):
                    if grid[ny][nx]["terrain"] == "water":
                        height_map[y][x] = min(height_map[y][x], 3)

    for y in range(height):
        for x in range(width):
            if grid[y][x]["terrain"] != "water":
                height_map[y][x] = max(1, height_map[y][x])

    # 5. Stone elevation boost
    def distance_to_grass(x, y):
        visited = set()
        queue = deque([(x, y, 0)])
        while queue:
            cx, cy, dist = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            if grid[cy][cx]["terrain"] == "grass":
                return dist
            for nx, ny in neighbors(cx, cy):
                queue.append((nx, ny, dist + 1))
        return 0

    for y in range(height):
        for x in range(width):
            if grid[y][x]["terrain"] == "stone":
                dist = distance_to_grass(x, y)
                height_map[y][x] += dist

    return height_map

def generate_structured_map(settings_file="gen-settings.json"):
    settings = load_generation_settings(settings_file)
    width = random.randint(6, 12)
    height = random.randint(6, 12)
    grid = make_empty_map(width, height)

    for terrain, config in settings.get("terrains", {}).items():
        spread_terrain(
            grid, terrain,
            attempts=config.get("attempts", 0),
            spread_chance=config.get("spread_chance", 0.5),
            min_distance=config.get("min_distance", 0)
        )

    height_settings = settings.get("height", {})
    height_map = generate_height_layer(grid, height_settings)


    for y in range(height):
        for x in range(width):
            grid[y][x]["height"] = height_map[y][x]
            grid[y][x]["trap"] = random.choices(TRAPS, weights=[0.8, 0.1, 0.1])[0]

    return {
        "width": width,
        "height": height,
        "grid": grid
    }

def save_map_to_json(map_data, filename="map.json"):
    with open(filename, "w") as f:
        json.dump(map_data, f, indent=2)

def load_map_from_json(filename="map.json"):
    with open(filename, "r") as f:
        return json.load(f)

def display_map_ASCII(map_data, layer="terrain"):
    symbols = {
        "water": "~",
        "stone": "#",
        "grass": ".",
        "wood": "T",
        "none": " "
    }

    for row in map_data["grid"]:
        line = ""
        for tile in row:
            value = tile.get(layer)
            if layer == "terrain":
                line += symbols.get(value, "?") + " "
            elif layer == "trap":
                line += value[0].upper() + " "
            elif layer == "height":
                line += f"{value:2d} "
        print(line)

def display_map(map_data, show_height=True, show_traps=True, show_creatures=True, output_file="map.png"):
    grid = map_data["grid"]
    tile_size = 32
    width = len(grid[0])
    height = len(grid)

    # Terrain base colors
    terrain_colors = {
        "water": (0, 105, 148),      # blue
        "stone": (120, 120, 120),    # grey
        "grass": (144, 238, 144),    # light green
        "wood": (34, 139, 34),       # dark green
        "none": (255, 255, 255),     # white
    }

    img = Image.new("RGB", (width * tile_size, height * tile_size), "white")
    draw = ImageDraw.Draw(img)

    # Use default font
    font = ImageFont.load_default()

    for y in range(height):
        for x in range(width):
            tile = grid[y][x]
            terrain = tile.get("terrain", "none")
            color = terrain_colors.get(terrain, (255, 0, 255))  # fallback magenta
            top_left = (x * tile_size, y * tile_size)
            bottom_right = ((x+1) * tile_size, (y+1) * tile_size)
            draw.rectangle([top_left, bottom_right], fill=color)

            # Trap label (top-left)
            if show_traps:
                trap = tile.get("trap")
                if trap and trap.lower() != "none":
                    draw.text((top_left[0] + 2, top_left[1] + 2), trap[0].upper(), fill="black", font=font)


            # Height label (bottom-right)
            if show_height:
                h = tile.get("height")
                if h is not None:
                    draw.text((bottom_right[0] - 12, bottom_right[1] - 12), str(h), fill="black", font=font)

            # Creature indicator (center)
            if show_creatures:
                creature = tile.get("creatures")
                if creature and creature.lower() != "none":
                    # Center of the tile
                    print(f"Creating creature {creature}")
                    center_x = top_left[0] + tile_size // 2
                    center_y = top_left[1] + tile_size // 2
                    radius = tile_size // 3
                    bbox = [
                        (center_x - radius, center_y - radius),
                        (center_x + radius, center_y + radius)
                    ]
                    draw.ellipse(bbox, fill="yellow", outline="black")

                    # First letter of the creature's name
                    initial = creature[0].upper()
                    bbox = draw.textbbox((0, 0), initial, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_pos = (
                        center_x - text_width // 2,
                        center_y - text_height // 2
                    )

                    draw.text(text_pos, initial, fill="black", font=font)


    img.save(output_file)
    print(f"Map saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate and display a new map")
    parser.add_argument("--save", action="store_true", help="Generate and save a new map to a file")
    parser.add_argument("--display", action="store_true", help="Display a map loaded from a file")

    parser.add_argument("--settings", type=str, default="gen-settings.json", help="Path to generation settings JSON")
    parser.add_argument("--map_file", type=str, default="map.json", help="Path to map JSON file")
    parser.add_argument("--layer", choices=["terrain", "trap", "height"], default="terrain", help="Layer to display")

    args = parser.parse_args()

    if args.generate:
        generated_map = generate_structured_map(args.settings)
        display_map_ASCII(generated_map, layer=args.layer)
        display_map(generated_map)

    elif args.save:
        generated_map = generate_structured_map(args.settings)
        display_map_ASCII(generated_map, layer=args.layer)
        save_map_to_json(generated_map, args.map_file)
        print(f"Map saved to {args.map_file}.")

    elif args.display:
        loaded_map = load_map_from_json(args.map_file)
        display_map_ASCII(loaded_map, layer=args.layer)
        display_map(loaded_map)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
