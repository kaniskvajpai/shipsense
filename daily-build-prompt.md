# ShipSense — Daily Build Prompt (30-Day Growth Plan)

Use this exact prompt every day during the 30-day growth plan. Only the day number changes.

---

```
Day [X] of the ShipSense 30-Day Growth Plan

Today is Day [X], continuing the same project across this ongoing chat.

Read 30-day-growth-plan.md and use it as the source of truth. Complete only
Day [X]'s listed milestone. Do not redesign the project or start a different
day's work.

Before writing any code, review everything built so far in this growth plan
(check git log and the current state of the repo) so today's work builds on
what actually exists, not what the plan assumed would exist.

Use only free tools, APIs, and hosting - no paid services without my explicit
approval.

Assume I have the same skill level as when I finished the original 10-day
capstone. Whenever I need to do something outside this chat (installing
something, configuring a service, deploying, etc.), give me exact step-by-step
instructions with real button/menu names and terminal commands. Wait for my
confirmation and a screenshot before continuing.

Prioritize implementation over explanation. Generate complete, copy-pasteable
files - no snippets or placeholders.

If today's milestone depends on a decision or file from an earlier growth-plan
day that wasn't actually completed, stop and tell me before proceeding.

When today's work is complete:
- Verify it works and didn't break anything built previously.
- Update any affected documentation (README.md, docs/ARCHITECTURE.md, 
  docs/SCHEMA.md, docs/API.md as relevant).
- Help me commit and push with a clear message referencing "Growth Plan Day [X]".
- Give a concise summary of what was completed and what Day [X+1] will build on.
```

---

## Notes for Using This Prompt

- Replace `[X]` with the actual day number (1–30) each time.
- If a day's milestone in `30-day-growth-plan.md` turns out to be too ambitious for one session, it's fine to split it — just say so explicitly and adjust the plan file accordingly, same discipline as the original 10-day capstone.
- Keep the same "flag before redesigning" discipline that worked well in the original capstone — if something in the growth plan conflicts with a decision already made, stop and ask rather than silently changing course.
