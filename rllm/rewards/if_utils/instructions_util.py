# coding=utf-8
"""Utility library for instruction following evaluation."""

import functools
import random
import re
from typing import List

#这是一个为"指令遵循评测"服务的小工具集：句子/词统计（带 NLTK→正则兜底与缓存）、句子切分、以及随机关键词生成
# 移除 NLTK 依赖，使用正则表达式分词

#一个很长的常用英文词列表，用于随机挑选关键词
WORD_LIST = ["western", "sentence", "signal", "dump", "spot", "opposite", "bottom", "potato", "administration", "working", "welcome", "morning", "good", "agency", "primary", "wish", "responsibility", "press", "problem", "president", "steal", "brush", "read", "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal", "comfortable", "region", "replacement", "performance", "mate", "walk", "medicine", "film", "thing", "rock", "tap", "total", "competition", "ease", "south", "establishment", "gather", "parking", "world", "plenty", "breath", "claim", "alcohol", "trade", "dear", "highlight", "street", "matter", "decision", "mess", "agreement", "studio", "coach", "assist", "brain", "wing", "style", "private", "top", "brown", "leg", "buy", "procedure", "method", "speed", "high", "company", "valuable", "pie", "analyst", "session", "pattern", "district", "pleasure", "dinner", "swimming", "joke", "order", "plate", "department", "motor", "cell", "spend", "cabinet", "difference", "power", "examination", "engine", "horse", "dimension", "pay", "toe", "curve", "literature", "bother", "fire", "possibility", "debate", "activity", "passage", "hello", "cycle", "background", "quiet", "author", "effect", "actor", "page", "bicycle", "error", "throat", "attack", "character", "phone", "tea", "increase", "outcome", "file", "specific", "inspector", "internal", "potential", "staff", "building", "employer", "shoe", "hand", "direction", "garden", "purchase", "interview", "study", "recognition", "member", "spiritual", "oven", "sandwich", "weird", "passenger", "particular", "response", "reaction", "size", "variation", "cancel", "candy", "exit", "guest", "condition", "fly", "price", "weakness", "convert", "hotel", "great", "mouth", "mind", "song", "sugar", "suspect", "telephone", "ear", "roof", "paint", "refrigerator", "organization", "jury", "reward", "engineering", "day", "possession", "crew", "bar", "road", "description", "celebration", "score", "mark", "letter", "shower", "suggestion", "sir", "luck", "national", "progress", "hall", "stroke", "theory", "offer", "story", "tax", "definition", "history", "ride", "medium", "opening", "glass", "elevator", "stomach", "question", "ability", "leading", "village", "computer", "city", "grand", "confidence", "candle", "priest", "recommendation", "point", "necessary", "body", "desk", "secret", "horror", "noise", "culture", "warning", "water", "round", "diet", "flower", "bus", "tough", "permission", "week", "prompt", "connection", "abuse", "height", "save", "corner", "border", "stress", "drive", "stop", "rip", "meal", "listen", "confusion", "girlfriend", "living", "relation", "significance", "plan", "creative", "atmosphere", "blame", "invite", "housing", "paper", "drink", "roll", "silver", "drunk", "age", "damage", "smoke", "environment", "pack", "savings", "influence", "tourist", "rain", "post", "sign", "grandmother", "run", "profit", "push", "clerk", "final", "wine", "swim", "pause", "stuff", "singer", "funeral", "average", "source", "scene", "tradition", "personal", "snow", "nobody", "distance", "sort", "sensitive", "animal", "major", "negotiation", "click", "mood", "period", "arrival", "expression", "holiday", "repeat", "dust", "closet", "gold", "bad", "sail", "combination", "clothes", "emphasis", "duty", "black", "step", "school", "jump", "document", "professional", "lip", "chemical", "front", "wake", "while", "inside", "watch", "row", "subject", "penalty", "balance", "possible", "adult", "aside", "sample", "appeal", "wedding", "depth", "king", "award", "wife", "blow", "site", "camp", "music", "safe", "gift", "fault", "guess", "act", "shame", "drama", "capital", "exam", "stupid", "record", "sound", "swing", "novel", "minimum", "ratio", "machine", "shape", "lead", "operation", "salary", "cloud", "affair", "hit", "chapter", "stage", "quantity", "access", "army", "chain", "traffic", "kick", "analysis", "airport", "time", "vacation", "philosophy", "ball", "chest", "thanks", "place", "mountain", "advertising", "red", "past", "rent", "return", "tour", "house", "construction", "net", "native", "war", "figure", "fee", "spray", "user", "dirt", "shot", "task", "stick", "friend", "software", "promotion", "interaction", "surround", "block", "purpose", "practice", "conflict", "routine", "requirement", "bonus", "hole", "state", "junior", "sweet", "catch", "tear", "fold", "wall", "editor", "life", "position", "pound", "respect", "bathroom", "coat", "script", "job", "teach", "birth", "view", "resolve", "theme", "employee", "doubt", "market", "education", "serve", "recover", "tone", "harm", "miss", "union", "understanding", "cow", "river", "association", "concept", "training", "recipe", "relationship", "reserve", "depression", "proof", "hair", "revenue", "independent", "lift", "assignment", "temporary", "amount", "loss", "edge", "track", "check", "rope", "estimate", "pollution", "stable", "message", "delivery", "perspective", "mirror", "assistant", "representative", "witness", "nature", "judge", "fruit", "tip", "devil", "town", "emergency", "upper", "drop", "stay", "human", "neck", "speaker", "network", "sing", "resist", "league", "trip", "signature", "lawyer", "importance", "gas", "choice", "engineer", "success", "part", "external", "worker", "simple", "quarter", "student", "heart", "pass", "spite", "shift", "rough", "lady", "grass", "community", "garage", "youth", "standard", "skirt", "promise", "blind", "television", "disease", "commission", "positive", "energy", "calm", "presence", "tune", "basis", "preference", "head", "common", "cut", "somewhere", "presentation", "current", "thought", "revolution", "effort", "master", "implement", "republic", "floor", "principle", "stranger", "shoulder", "grade", "button", "tennis", "police", "collection", "account", "register", "glove", "divide", "professor", "chair", "priority", "combine", "peace", "extension", "maybe", "evening", "frame", "sister", "wave", "code", "application", "mouse", "match", "counter", "bottle", "half", "cheek", "resolution", "back", "knowledge", "make", "discussion", "screw", "length", "accident", "battle", "dress", "knee", "log", "package", "turn", "hearing", "newspaper", "layer", "wealth", "profile", "imagination", "answer", "weekend", "teacher", "appearance", "meet", "bike", "rise", "belt", "crash", "bowl", "equivalent", "support", "image", "poem", "risk", "excitement", "remote", "secretary", "public", "produce", "plane", "display", "money", "sand", "situation", "punch", "customer", "title", "shake", "mortgage", "option", "number", "pop", "window", "extent", "nothing", "experience", "opinion", "departure", "dance", "indication", "boy", "material", "band", "leader", "sun", "beautiful", "muscle", "farmer", "variety", "fat", "handle", "director", "opportunity", "calendar", "outside", "pace", "bath", "fish", "consequence", "put", "owner", "go", "doctor", "information", "share", "hurt", "protection", "career", "finance", "force", "golf", "garbage", "aspect", "kid", "food", "boot", "milk", "respond", "objective", "reality", "raw", "ring", "mall", "one", "impact", "area", "news", "international", "series", "impress", "mother", "shelter", "strike", "loan", "month", "seat", "anything", "entertainment", "familiar", "clue", "year", "glad", "supermarket", "natural", "god", "cost", "conversation", "tie", "ruin", "comfort", "earth", "storm", "percentage", "assistance", "budget", "strength", "beginning", "sleep", "other", "young", "unit", "fill", "store", "desire", "hide", "value", "cup", "maintenance", "nurse", "function", "tower", "role", "class", "camera", "database", "panic", "nation", "basket", "ice", "art", "spirit", "chart", "exchange", "feedback", "statement", "reputation", "search", "hunt", "exercise", "nasty", "notice", "male", "yard", "annual", "collar", "date", "platform", "plant", "fortune", "passion", "friendship", "spread", "cancer", "ticket", "attitude", "island", "active", "object", "service", "buyer", "bite", "card", "face", "steak", "proposal", "patient", "heat", "rule", "resident", "broad", "politics", "west", "knife", "expert", "girl", "design", "salt", "baseball", "grab", "inspection", "cousin", "couple", "magazine", "cook", "dependent", "security", "chicken", "version", "currency", "ladder", "scheme", "kitchen", "employment", "local", "attention", "manager", "fact", "cover", "sad", "guard", "relative", "county", "rate", "lunch", "program", "initiative", "gear", "bridge", "breast", "talk", "dish", "guarantee", "beer", "vehicle", "reception", "woman", "substance", "copy", "lecture", "advantage", "park", "cold", "death", "mix", "hold", "scale", "tomorrow", "blood", "request", "green", "cookie", "church", "strip", "forever", "beyond", "debt", "tackle", "wash", "following", "feel", "maximum", "sector", "sea", "property", "economics", "menu", "bench", "try", "language", "start", "call", "solid", "address", "income", "foot", "senior", "honey", "few", "mixture", "cash", "grocery", "link", "map", "form", "factor", "pot", "model", "writer", "farm", "winter", "skill", "anywhere", "birthday", "policy", "release", "husband", "lab", "hurry", "mail", "equipment", "sink", "pair", "driver", "consideration", "leather", "skin", "blue", "boat", "sale", "brick", "two", "feed", "square", "dot", "rush", "dream", "location", "afternoon", "manufacturer", "control", "occasion", "trouble", "introduction", "advice", "bet", "eat", "kill", "category", "manner", "office", "estate", "pride", "awareness", "slip", "crack", "client", "nail", "shoot", "membership", "soft", "anybody", "web", "official", "individual", "pizza", "interest", "bag", "spell", "profession", "queen", "deal", "resource", "ship", "guy", "chocolate", "joint", "formal", "upstairs", "car", "resort", "abroad", "dealer", "associate", "finger", "surgery", "comment", "team", "detail", "crazy", "path", "tale", "initial", "arm", "radio", "demand", "single", "draw", "yellow", "contest", "piece", "quote", "pull", "commercial", "shirt", "contribution", "cream", "channel", "suit", "discipline", "instruction", "concert", "speech", "low", "effective", "hang", "scratch", "industry", "breakfast", "lay", "join", "metal", "bedroom", "minute", "product", "rest", "temperature", "many", "give", "argument", "print", "purple", "laugh", "health", "credit", "investment", "sell", "setting", "lesson", "egg", "middle", "marriage", "level", "evidence", "phrase", "love", "self", "benefit", "guidance", "affect", "you", "dad", "anxiety", "special", "boyfriend", "test", "blank", "payment", "soup", "obligation", "reply", "smile", "deep", "complaint", "addition", "review", "box", "towel", "minor", "fun", "soil", "issue", "cigarette", "internet", "gain", "tell", "entry", "spare", "incident", "family", "refuse", "branch", "can", "pen", "grandfather", "constant", "tank", "uncle", "climate", "ground", "volume", "communication", "kind", "poet", "child", "screen", "mine", "quit", "gene", "lack", "charity", "memory", "tooth", "fear", "mention", "marketing", "reveal", "reason", "court", "season", "freedom", "land", "sport", "audience", "classroom", "law", "hook", "win", "carry", "eye", "smell", "distribution", "research", "country", "dare", "hope", "whereas", "stretch", "library", "delay", "college", "plastic", "book", "present", "use", "worry", "champion", "goal", "economy", "march", "election", "reflection", "midnight", "slide", "inflation", "action", "challenge", "guitar", "coast", "apple", "campaign", "field", "jacket", "sense", "way", "visual", "remove", "weather", "trash", "cable", "regret", "buddy", "beach", "historian", "courage", "sympathy", "truck", "tension", "permit", "nose", "bed", "son", "person", "base", "meat", "usual", "air", "meeting", "worth", "game", "independence", "physical", "brief", "play", "raise", "board", "key", "writing", "pick", "command", "party", "yesterday", "spring", "candidate", "physics", "university", "concern", "development", "change", "string", "target", "instance", "room", "bitter", "bird", "football", "normal", "split", "impression", "wood", "long", "meaning", "stock", "cap", "leadership", "media", "ambition", "fishing", "essay", "salad", "repair", "today", "designer", "night", "bank", "drawing", "inevitable", "phase", "vast", "chip", "anger", "switch", "cry", "twist", "personality", "attempt", "storage", "being", "preparation", "bat", "selection", "white", "technology", "contract", "side", "section", "station", "till", "structure", "tongue", "taste", "truth", "difficulty", "group", "limit", "main", "move", "feeling", "light", "example", "mission", "might", "wait", "wheel", "shop", "host", "classic", "alternative", "cause", "agent", "consist", "table", "airline", "text", "pool", "craft", "range", "fuel", "tool", "partner", "load", "entrance", "deposit", "hate", "article", "video", "summer", "feature", "extreme", "mobile", "hospital", "flight", "fall", "pension", "piano", "fail", "result", "rub", "gap", "system", "report", "suck", "ordinary", "wind", "nerve", "ask", "shine", "note", "line", "mom", "perception", "brother", "reference", "bend", "charge", "treat", "trick", "term", "homework", "bake", "bid", "status", "project", "strategy", "orange", "let", "enthusiasm", "parent", "concentrate", "device", "travel", "poetry", "business", "society", "kiss", "end", "vegetable", "employ", "schedule", "hour", "brave", "focus", "process", "movie", "illegal", "general", "coffee", "ad", "highway", "chemistry", "psychology", "hire", "bell", "conference", "relief", "show", "neat", "funny", "weight", "quality", "club", "daughter", "zone", "touch", "tonight", "shock", "burn", "excuse", "name", "survey", "landscape", "advance", "satisfaction", "bread", "disaster", "item", "hat", "prior", "shopping", "visit", "east", "photo", "home", "idea", "father", "comparison", "cat", "pipe", "winner", "count", "lake", "fight", "prize", "foundation", "dog", "keep", "ideal", "fan", "struggle", "peak", "safety", "solution", "hell", "conclusion", "population", "strain", "alarm", "measurement", "second", "train", "race", "due", "insurance", "boss", "tree", "monitor", "sick", "course", "drag", "appointment", "slice", "still", "care", "patience", "rich", "escape", "emotion", "royal", "female", "childhood", "government", "picture", "will", "sock", "big", "gate", "oil", "cross", "pin", "improvement", "championship", "silly", "help", "sky", "pitch", "man", "diamond", "most", "transition", "work", "science", "committee", "moment", "fix", "teaching", "dig", "specialist", "complex", "guide", "people", "dead", "voice", "original", "break", "topic", "data", "degree", "reading", "recording", "bunch", "reach", "judgment", "lie", "regular", "set", "painting", "mode", "list", "player", "bear", "north", "wonder", "carpet", "heavy", "officer", "negative", "clock", "unique", "baby", "pain", "assumption", "disk", "iron", "bill", "drawer", "look", "double", "mistake", "finish", "future", "brilliant", "contact", "math", "rice", "leave", "restaurant", "discount", "sex", "virus", "bit", "trust", "event", "wear", "juice", "failure", "bug", "context", "mud", "whole", "wrap", "intention", "draft", "pressure", "cake", "dark", "explanation", "space", "angle", "word", "efficiency", "management", "habit", "star", "chance", "finding", "transportation", "stand", "criticism", "flow", "door", "injury", "insect", "surprise", "apartment"]

# ISO 639-1 codes to language names.
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish", 
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ms": "Malay",
    "ro": "Romanian",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "hu": "Hungarian",
    "cs": "Czech",
    "sk": "Slovak",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "ga": "Irish",
    "cy": "Welsh",
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    "is": "Icelandic",
    "mk": "Macedonian",
    "sq": "Albanian",
    "be": "Belarusian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "uz": "Uzbek",
    "mn": "Mongolian",
    "hy": "Armenian",
    "ka": "Georgian",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "ka": "Georgian",
    "am": "Amharic",
    "sw": "Swahili",
    "zu": "Zulu",
    "af": "Afrikaans",
    "ig": "Igbo",
    "yo": "Yoruba",
    "ha": "Hausa",
    "so": "Somali",
    "rw": "Kinyarwanda",
    "ny": "Chichewa",
    "mg": "Malagasy",
    "eo": "Esperanto",
    "la": "Latin",
    "jw": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
    "haw": "Hawaiian",
    "mi": "Maori",
    "sm": "Samoan",
    "to": "Tongan",
    "fj": "Fijian",
    "ty": "Tahitian",
    "co": "Corsican",
    "fy": "Frisian",
    "gd": "Scottish Gaelic",
    "lb": "Luxembourgish",
    "rm": "Romansh",
    "br": "Breton",
    "kw": "Cornish",
    "gv": "Manx",
    "fo": "Faroese",
    "kl": "Greenlandic",
    "se": "Northern Sami",
    "yi": "Yiddish",
    "he": "Hebrew",
    "ar": "Arabic",
    "fa": "Persian",
    "ur": "Urdu",
    "ps": "Pashto",
    "sd": "Sindhi",
    "ku": "Kurdish",
    "ckb": "Sorani Kurdish"
}


@functools.lru_cache(maxsize=None)
@functools.lru_cache(maxsize=None)
def count_sentences(text: str) -> int:
    """Count sentences in text using regex."""
    # Simple sentence counting based on punctuation
    sentences = re.split(r'[.!?]+', text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

@functools.lru_cache(maxsize=None) 
def count_words(text: str) -> int:
    """Count words in text."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Simple sentence splitting based on punctuation
    sentences = re.split(r'[.!?]+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_keywords(num_keywords: int) -> List[str]:
    """Generate random keywords from the word list."""
    return random.sample(WORD_LIST, min(num_keywords, len(WORD_LIST)))