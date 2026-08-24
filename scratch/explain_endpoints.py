@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability Agent"])
async def explain_proof(request: ExplainRequest):
    """Sync proof explanation — LLM or template fallback."""
    trace_text = "\n".join(
        f"Step {i + 1}: [{s.rule_id}] {s.fired_rule_repr} → New facts: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(
            request.auxiliary_constructions
        )

    from rag_agent.llm_factory import get_llm

    llm = get_llm(temperature=0.3)

    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=_build_explain_system_prompt(request.goal_reached)),
                HumanMessage(content=f"Query: '{request.query}'\n\nProof Trace:\n{trace_text or 'No rules triggered.'}{aux_text}")
            ]
            response = llm.invoke(messages)
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return ExplainResponse(explanation=content, structured=True)
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)

    # Template fallback
    parts = []
    if request.goal_reached:
        parts = [
            "# Lời giải Hình học\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Mục tiêu của bài toán đã được chứng minh thành công.\n\n",
        ]
        if request.auxiliary_constructions:
            parts.append(
                f"**Đường phụ đã dựng:** {', '.join(request.auxiliary_constructions)}\n\n"
            )

        parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        parts.append("## Các bước suy luận\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append("## ✓ Kết luận\nBài toán đã được chứng minh trọn vẹn.\n")
    else:
        parts = [
            "# ⚠️ Chưa hoàn tất chứng minh\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Hệ thống chưa tìm thấy chuỗi định lý kết nối trực tiếp đến mục tiêu.\n\n",
        ]
        parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        if not request.execution_trace:
            parts.append("Chưa có định lý nào được kích hoạt từ giả thiết ban đầu.\n")
        for i, step in enumerate(request.execution_trace):
            parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        parts.append("</details>\n\n")
        parts.append(
            "## Gợi ý\nCó thể cần bổ sung thêm giả thiết hoặc kẻ thêm đường phụ để tạo cầu nối suy luận.\n"
        )

    return ExplainResponse(explanation="".join(parts), structured=False)


@app.post("/api/explain/stream", tags=["Explainability Agent"])
async def explain_proof_stream(request: ExplainRequest):
    """Streaming proof explanation — LLM real-time or template chunk stream."""
    trace_text = "\n".join(
        f"Step {i + 1}: [{s.rule_id}] {s.fired_rule_repr} → New: {s.new_facts}"
        for i, s in enumerate(request.execution_trace)
    )
    aux_text = ""
    if request.auxiliary_constructions:
        aux_text = "\nAuxiliary constructions added: " + ", ".join(
            request.auxiliary_constructions
        )

    # Prepare template fallback parts
    if request.goal_reached:
        fallback_parts = [
            "# Lời giải Hình học\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Mục tiêu của bài toán đã được chứng minh thành công.\n\n",
        ]
        if request.auxiliary_constructions:
            fallback_parts.append(
                f"**Đường phụ đã dựng:** {', '.join(request.auxiliary_constructions)}\n\n"
            )
        fallback_parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        fallback_parts.append("## Các bước suy luận\n")
        for i, step in enumerate(request.execution_trace):
            fallback_parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            fallback_parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            fallback_parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        fallback_parts.append("</details>\n\n")
        fallback_parts.append("## ✓ Kết luận\nBài toán đã được chứng minh trọn vẹn.\n")
    else:
        fallback_parts = [
            "# ⚠️ Chưa hoàn tất chứng minh\n\n",
            f"**Đề bài:** *{request.query}*\n\n",
            "Hệ thống chưa tìm thấy chuỗi định lý kết nối trực tiếp đến mục tiêu.\n\n",
        ]
        fallback_parts.append(
            "<details>\n<summary><b>📋 Chi tiết các bước suy luận hệ thống (Deduction Steps)</b></summary>\n\n"
        )
        if not request.execution_trace:
            fallback_parts.append("Chưa có định lý nào được kích hoạt từ giả thiết ban đầu.\n")
        for i, step in enumerate(request.execution_trace):
            fallback_parts.append(f"### Bước {i + 1}: `{step.rule_id}`\n")
            fallback_parts.append(f"- **Định lý:** `{step.fired_rule_repr}`\n")
            fallback_parts.append(f"- **Tính chất mới:** `{', '.join(step.new_facts)}`\n\n")
        fallback_parts.append("</details>\n\n")
        fallback_parts.append(
            "## Gợi ý\nCó thể cần bổ sung thêm giả thiết hoặc kẻ thêm đường phụ để tạo cầu nối suy luận.\n"
        )

    from rag_agent.llm_factory import get_llm

    llm = get_llm(temperature=0.3)

    async def stream_generator():
        yielded_any = False
        if llm:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage

                messages = [
                    SystemMessage(content=_build_explain_system_prompt(request.goal_reached)),
                    HumanMessage(content=f"Query: '{request.query}'\n\nProof Trace:\n{trace_text or 'No rules triggered.'}{aux_text}")
                ]
                async for chunk in llm.astream(messages):
                    content = chunk.content
                    if not content:
                        continue
                    text = content if isinstance(content, str) else str(content)
                    if text:
                        yielded_any = True
                        yield text
            except Exception as e:
                logger.warning("Streaming LLM explanation failed: %s — falling back to template", e)

        if not yielded_any:
            for part in fallback_parts:
                yield part
                await asyncio.sleep(0.01)

    return StreamingResponse(stream_generator(), media_type="text/plain")
