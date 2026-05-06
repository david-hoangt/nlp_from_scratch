# Positional Encoding Rewrite Plan

## Locked Decisions

### Keep

- Giữ full implementations trong blog để nhất quán với series "building from scratch in PyTorch".
- Giữ progression hiện tại: Absolute -> Relative/ALiBi -> RoPE -> p-RoPE -> NoPE.
- Giữ hooks giữa các section để bài có narrative pull.
- Giữ p-RoPE và NoPE trong cùng bài, không tách ra post riêng.

### Change

- Đưa compact decision table lên ngay sau intro + pipeline diagram.
- Nén mạnh section Relative PE: Shaw là cơ chế chính; Transformer-XL và T5 chỉ là short follow-up.
- Gộp notebook CTA còn 2 lần: một lần gần đầu bài, một lần sau RoPE hoặc ở cuối bài.
- Vary phrasing của hooks thay vì lặp đi lặp lại "What if...".
- Scan nhanh lại claims và alt text trong pass cuối.

## Rewrite Goals

1. Cho reader roadmap sớm thay vì bắt họ đọc gần hết bài mới biết "nên dùng cái nào".
2. Giảm cognitive spike lớn nhất ở Relative PE mà không làm mất chiều sâu kỹ thuật.
3. Giữ promise của series: code là content, không phải appendix.
4. Preserve insight cuối bài: explicit positional information đi từ everywhere -> where needed -> maybe nowhere.

## Target Structure

1. Intro
2. Pipeline diagram
3. Compact decision table
4. Absolute Positional Encoding
5. Relative Positional Encoding
6. ALiBi
7. RoPE
8. Bonus / Frontier: p-RoPE
9. Frontier: NoPE
10. Evolution at a Glance
11. Which PE Should You Use?
12. References

## Section-by-Section Plan

### 1. Intro + Roadmap

- Giữ hook hiện tại và pipeline diagram.
- Ngay sau diagram, thêm một compact table để reader có bản đồ sớm.
- Thêm một câu định hướng ngắn kiểu: "If you only need the practical answer, read this table first, then come back for the mechanics."
- Đặt CTA notebook đầu tiên ở đây, thay vì rải nhiều chỗ giữa bài.

### 2. Decision Table Placement

Table nên thật ngắn, đủ để orient reader chứ không thay phần thân bài.

Suggested columns:

- Scheme
- Injects where
- Best fit
- Main limitation

Suggested rows:

- Absolute PE | input embeddings | encoders, pedagogy, classic baselines | weak direct notion of relative distance
- Relative PE | attention scores | encoder work, explicit distance modeling | extra complexity, less common in current decoder LLMs
- ALiBi | attention scores | length extrapolation with minimal params | hardcoded recency bias
- RoPE | Q/K vectors | current default for decoder-only LLMs | long-context degradation in low-frequency bands
- p-RoPE / NoPE | selective or no explicit PE | frontier long-context decoder designs | less standardized, more architecture-specific

### 3. Absolute PE

- Giữ phần này gần như nguyên vẹn.
- Chỉ tighten những đoạn giải thích lặp ý giữa sinusoidal và learned embeddings.
- Kết section bằng một bridge rõ: absolute methods stamp position identity, but not distance directly.

### 4. Relative PE

- Giữ Shaw et al. làm main path.
- Giữ clipping matrix vì đây là artifact dễ hiểu nhất của whole family.
- Giữ full `RelativePositionAttention` implementation để nhất quán với series.
- Cắt bớt prose giải thích xung quanh code khoảng 25-35%.
- Gộp Transformer-XL và T5 thành một subsection ngắn kiểu "same family, different engineering trade-offs".

Target takeaway của section này:

- Reader nhớ một ý chính: score-based methods inject distance into logits.

### 5. ALiBi

- Giữ full section và full code.
- Tighten phần mở đầu và bias-matrix explanation nếu đang lặp lại ý đã có ở Relative PE.
- Nhấn rõ hơn vai trò của ALiBi như một simpler score-based branch, không phải một detour hoàn toàn mới.

### 6. RoPE

- Giữ đây là section dài và quan trọng nhất.
- Giữ rotation mechanism, relative-position property, two-band finding, context extension, và full implementation.
- CTA notebook thứ hai đặt sau section này là hợp lý nhất, vì đây là payoff point mạnh nhất của bài.
- Kết section bằng một practical line rõ ràng: nếu đang build decoder-only LLM hiện nay, đây là baseline mặc định.

### 7. p-RoPE and NoPE

- Không cắt thành micro-section.
- Giữ cả hai như current frontier continuation của RoPE.
- Thêm một signpost trước p-RoPE kiểu: "From this point on, the question changes from how to encode position to how much position you actually need."
- Giữ tag `Bonus` cho p-RoPE.
- Với NoPE, có thể thêm label `Frontier` ở opening paragraph để reader biết đây là phần advanced but relevant.

Target takeaway của two-section block này:

- Câu hỏi không còn là "PE nào đúng nhất" mà là "model cần bao nhiêu explicit position information".

### 8. Hooks Between Sections

- Giữ hooks.
- Không bỏ narrative transitions.
- Chỉ vary phrasing để tránh pattern fatigue.

Examples of alternative hook styles:

- "That solves position identity, but not distance."
- "A learned bias helps, but it still lives outside Q and K."
- "Push the position signal into the vectors themselves and the dot product changes character."
- "Once rotation works, the next question is whether every dimension needs it."
- "If some dimensions survive without explicit position, can a whole layer do the same?"

### 9. CTA Consolidation

Final rule:

- CTA 1: near the top, after intro + roadmap table.
- CTA 2: after RoPE or at the end.

Remove the in-section CTA repetitions from Absolute, ALiBi, RoPE, and NoPE sections.

### 10. Final Cleanup Pass

- Scan claims and keep them qualified where needed.
- Add meaningful alt text for static images.
- Add one short decision summary at the end of each major section:
	- when this scheme is useful
	- why it wins
	- why it loses

## Non-Goals

- Không đổi bài thành notebook companion only.
- Không rút full implementations xuống thành snippets 5-10 lines.
- Không cắt bài ở RoPE.
- Không bỏ p-RoPE / NoPE khỏi bài.
- Không bỏ hooks hoàn toàn.

## Execution Order

1. Add decision table after intro.
2. Rewrite Relative PE to reduce weight.
3. Consolidate CTAs.
4. Vary section hooks.
5. Add frontier signposting before p-RoPE / NoPE.
6. Run final claim and alt-text scan.

## Ready-to-Implement Summary

This rewrite keeps the article as a code-first long-form post, but fixes the reader experience at the structural level:

- roadmap earlier
- Relative PE lighter
- fewer CTA interruptions
- same full-code value proposition
- same RoPE -> p-RoPE -> NoPE insight arc