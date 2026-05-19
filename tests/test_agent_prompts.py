from pathlib import Path

import pytest

from tr_agent.agent.prompts import news_section, rules_section


class TestNewsSection:
    def test_empty_when_no_items(self):
        assert news_section([]) == ""

    def test_formats_single_headline(self):
        items = [{"title": "NVDA beats estimates", "publisher": "Reuters", "age_str": "2h ago"}]
        result = news_section(items)
        assert "Recent news:" in result
        assert "NVDA beats estimates" in result
        assert "Reuters" in result
        assert "2h ago" in result

    def test_formats_multiple_headlines(self):
        items = [
            {"title": "Headline A", "publisher": "Bloomberg", "age_str": "1h ago"},
            {"title": "Headline B", "publisher": "WSJ", "age_str": "5h ago"},
        ]
        result = news_section(items)
        assert "Headline A" in result
        assert "Headline B" in result

    def test_handles_missing_publisher(self):
        items = [{"title": "Some news", "publisher": "", "age_str": "3h ago"}]
        result = news_section(items)
        assert "Some news" in result
        assert "3h ago" in result


class TestRulesSection:
    def test_empty_when_file_does_not_exist(self, tmp_path):
        path = tmp_path / "nonexistent_rules.md"
        assert rules_section(path) == ""

    def test_returns_content_when_file_exists(self, tmp_path):
        rules_file = tmp_path / "llm_rules.md"
        rules_file.write_text("## Learned Rules\n- RSI < 30 wins 68% of the time")
        result = rules_section(rules_file)
        assert "Learned Rules" in result
        assert "RSI < 30" in result

    def test_truncates_long_content(self, tmp_path):
        rules_file = tmp_path / "llm_rules.md"
        rules_file.write_text("x" * 3000)
        result = rules_section(rules_file)
        assert len(result) < 2100  # 2000 content + header + truncation notice
        assert "truncated" in result

    def test_empty_when_file_is_blank(self, tmp_path):
        rules_file = tmp_path / "llm_rules.md"
        rules_file.write_text("   \n  ")
        assert rules_section(rules_file) == ""

    def test_includes_prefix_label(self, tmp_path):
        rules_file = tmp_path / "llm_rules.md"
        rules_file.write_text("## My Rules\n- Do this")
        result = rules_section(rules_file)
        assert result.startswith("Learned rules (from past performance):")
