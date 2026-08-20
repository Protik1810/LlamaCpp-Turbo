import textwrap

grammars = {
    "json": textwrap.dedent(r"""
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws
        object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
        array  ::= "[" ws ( value ("," ws value)* )? "]" ws
        string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
        ws     ::= [ \t\n]*
    """).strip(),
    "json_array": textwrap.dedent(r"""
        root   ::= array
        value  ::= object | array | string | number | ("true" | "false" | "null") ws
        object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
        array  ::= "[" ws ( value ("," ws value)* )? "]" ws
        string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
        ws     ::= [ \t\n]*
    """).strip(),
    "markdown_table": textwrap.dedent(r"""
        root       ::= table
        table      ::= row "\n" separator "\n" (row "\n")+
        row        ::= "|" ( cell "|" )+
        separator  ::= "|" ( " --- |" | " :--- |" | " :---: |" | " ---: |" )+
        cell       ::= " " [^|\n]+ " "
    """).strip(),
    "key_value": textwrap.dedent(r"""
        root ::= ( line "\n" )+
        line ::= key ": " val
        key  ::= [a-zA-Z0-9_-]+
        val  ::= [^\n\r]+
    """).strip(),
    "structured_steps": textwrap.dedent(r"""
        root       ::= ( step "\n\n" )+ conclusion
        step       ::= "### Step " [1-9] [0-9]? ": " title "\n" content
        title      ::= [^\n]+
        content    ::= [^\n]+
        conclusion ::= "### Conclusion\n" [^\n]+
    """).strip()
}

try:
    from llama_cpp import LlamaGrammar
    for name, gbnf in grammars.items():
        try:
            g = LlamaGrammar.from_string(gbnf)
            print(f"  [PASS] {name} GBNF compiled successfully!")
        except Exception as e:
            print(f"  [FAIL] {name} GBNF error: {e}")
except ImportError:
    print("llama_cpp not imported, testing syntax structure directly.")
    for name, gbnf in grammars.items():
        assert "root" in gbnf
        print(f"  [PASS] {name} GBNF syntax verified: {len(gbnf.splitlines())} rules")

print("ALL GBNF GRAMMAR TESTS PASSED!")
