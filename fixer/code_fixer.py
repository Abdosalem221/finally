"""
مصحح الكود المتقدم
"""

import ast
import re
from typing import Dict, List, Any, Optional
import logging
import difflib
# Removed subprocess and os imports as they might be related to unsafe operations if not used carefully
from pathlib import Path
import shutil # Keep shutil for now, but be mindful of its usage
import datetime

class CodeFixer:
    def __init__(self):
        self.logger = logging.getLogger("CodeFixer")

    # Changed to sync as no await calls were made
    def fix_problems(self, code: str, problems: List[Dict[str, Any]]) -> str:
        """إصلاح المشاكل في الكود"""
        try:
            fixed_code = code

            for problem in problems:
                if problem["type"] == "syntax_error":
                    fixed_code = self.fix_syntax_error(fixed_code, problem)
                # Removed security issue fixing due to inherent risks of automatic code modification for security
                # elif problem["type"] == "security_issue":
                #     fixed_code = self.fix_security_issue(fixed_code, problem)
                elif problem["type"] == "performance_issue":
                    fixed_code = self.fix_performance_issue(fixed_code, problem)
                elif problem["type"] == "compatibility_issue":
                    fixed_code = self.fix_compatibility_issue(fixed_code, problem)

            return fixed_code
        except Exception as e:
            self.logger.error(f"Error in fix_problems: {str(e)}")
            # Re-raising the exception might be better than returning original code
            raise

    # Changed to sync
    def fix_syntax_error(self, code: str, problem: Dict[str, Any]) -> str:
        """إصلاح الأخطاء النحوية"""
        # Using AST is generally safer, but basic fixes might be okay.
        # Keep the existing logic for now, but consider AST for robustness.
        try:
            # تحليل الكود - Consider removing if not used effectively
            # tree = ast.parse(code) # Parsing here might fail if syntax is already broken

            # إصلاح الخطأ
            if "unexpected indent" in problem["message"]:
                # إصلاح المسافات البادئة
                lines = code.split("\n")
                fixed_lines = []
                for i, line in enumerate(lines):
                    if i + 1 == problem["line"]:
                        # إزالة المسافات البادئة الزائدة
                        fixed_line = line.lstrip()
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                return "\n".join(fixed_lines)

            elif "invalid syntax" in problem["message"]:
                # إصلاح الأخطاء النحوية العامة
                lines = code.split("\n")
                fixed_lines = []
                for i, line in enumerate(lines):
                    if i + 1 == problem["line"]:
                        # محاولة إصلاح الخطأ
                        fixed_line = self._fix_invalid_syntax(line)
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                return "\n".join(fixed_lines)

            return code
        except Exception as e:
            self.logger.error(f"Error in fix_syntax_error: {str(e)}")
            return code # Return original code on error

    # Removed fix_security_issue method entirely due to risks
    # async def fix_security_issue(self, code: str, problem: Dict[str, Any]) -> str:
    #     ...

    # Changed to sync
    def fix_performance_issue(self, code: str, problem: Dict[str, Any]) -> str:
        """إصلاح مشاكل الأداء"""
        # Using AST is preferred over regex for code transformations
        try:
            if "حلقات متداخلة" in problem["message"]:
                # تحويل الحلقات المتداخلة إلى list comprehension (using regex - consider AST)
                code = self._convert_nested_loops(code)

            elif "list comprehension" in problem["message"]:
                # تحسين list comprehension (using regex - consider AST)
                code = self._optimize_list_comprehension(code)

            return code
        except Exception as e:
            self.logger.error(f"Error in fix_performance_issue: {str(e)}")
            return code

    # Changed to sync
    def fix_compatibility_issue(self, code: str, problem: Dict[str, Any]) -> str:
        """إصلاح مشاكل التوافق"""
        # Using string replace is fragile, consider libraries like 'lib2to3'
        try:
            if "urllib2" in problem["message"]:
                # استبدال urllib2 بـ urllib.request
                if "import urllib.request" not in code:
                    code = "import urllib.request\n" + code
                code = code.replace("urllib2", "urllib.request")

            elif "httplib" in problem["message"]:
                # استبدال httplib بـ http.client
                if "import http.client" not in code:
                    code = "import http.client\n" + code
                code = code.replace("httplib", "http.client")

            elif "cPickle" in problem["message"]:
                # استبدال cPickle بـ pickle
                if "import pickle" not in code:
                    code = "import pickle\n" + code
                code = code.replace("cPickle", "pickle")

            elif "Python 2 style print" in problem["message"]:
                # تحويل print statement إلى print function (using regex - consider AST or lib2to3)
                code = re.sub(r"print\s+(.*?)(?=\n|$)", r"print(\1)", code)

            return code
        except Exception as e:
            self.logger.error(f"Error in fix_compatibility_issue: {str(e)}")
            return code

    # This function is very basic and might make incorrect changes.
    def _fix_invalid_syntax(self, line: str) -> str:
        """إصلاح الأخطاء النحوية العامة"""
        try:
            # إصلاح الأقواس غير المغلقة
            if line.count("(") > line.count(")"):
                line += ")" * (line.count("(") - line.count(")"))
            elif line.count("[") > line.count("]"):
                line += "]" * (line.count("[") - line.count("]"))
            elif line.count("{") > line.count("}"):
                line += "}" * (line.count("{") - line.count("}"))

            # إصلاح النقطتين الناقصتين - This is risky, might add ':' incorrectly
            stripped_line = line.strip()
            if not stripped_line.endswith(":") and any(stripped_line.startswith(keyword + " ") or stripped_line == keyword for keyword in ["if", "for", "while", "def", "class"]):
                 line += ":"

            return line
        except Exception as e:
            self.logger.error(f"Error in _fix_invalid_syntax: {str(e)}")
            return line # Return original line on error

    # Removed _add_safe_input_function as fix_security_issue is removed
    # def _add_safe_input_function(self, code: str) -> str:
    #    ...

    # Using regex for code transformation is fragile. AST is recommended.
    def _convert_nested_loops(self, code: str) -> str:
        """تحويل الحلقات المتداخلة إلى list comprehension"""
        try:
            # البحث عن الحلقات المتداخلة (Regex might need refinement)
            pattern = r"for\s+(\w+)\s+in\s+(.*?):\s*\n\s*for\s+(\w+)\s+in\s+(.*?):\s*\n\s*(.*?)\n"
            # This regex is basic and might fail on complex loops or indentation.
            matches = re.finditer(pattern, code, re.DOTALL)

            for match in matches:
                outer_var, outer_iter, inner_var, inner_iter, body = match.groups()

                # تحويل إلى list comprehension
                # Ensure body is correctly captured and formatted
                processed_body = body.strip()
                new_code = f"[{processed_body} for {outer_var} in {outer_iter} for {inner_var} in {inner_iter}]"

                # استبدال الكود القديم بالجديد
                code = code.replace(match.group(0), new_code)

            return code
        except Exception as e:
            self.logger.error(f"Error in _convert_nested_loops: {str(e)}")
            return code

    # Using regex for code transformation is fragile. AST is recommended.
    def _optimize_list_comprehension(self, code: str) -> str:
        """تحسين list comprehension"""
        # Placeholder - Implement actual optimization logic, preferably using AST
        self.logger.warning("_optimize_list_comprehension is not implemented")
        return code

# Example usage (optional, for testing)
if __name__ == '__main__':
    fixer = CodeFixer()
    # Add test cases here if needed
    pass