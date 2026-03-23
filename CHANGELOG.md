# Changelog

---

# 1.0.6 — 2026-03-23

More robust accumulation of events + calculation of distance between visits and relapses.

### Changed
- Functions MSprog(), value_milestone(), separate_ri_ra()

---

# 1.0.5 — 2026-03-17

Date/value columns & 'event_type'/'CDW_type' columns in results, outcome argument values ('custom').

### Changed
- Functions MSprog(), value_milestone(), separate_ri_ra(), compute_delta(), is_event()

---

# 1.0.4 — 2026-02-05

Included date_format argument.

### Changed
- Functions MSprog(), value_milestone(), separate_ri_ra()

---

# 1.0.3 — 2026-01-19

Changed valid values for event argument.

### Changed
- Functions MSprog(), is_event()

---

# 1.0.2 — 2025-09-22

Saved toy data as csv instead of xlsx to avoid dependency on openpyxl.

### Changed
- Function load_toy_data()

---

# 1.0.1 — 2025-09-22

Align with msprog R package.

### Changed
- Functions MSprog(), separate_ri_ra(), value_milestone(), is_event(), load_toy_data()

---

# 1.0.0 — 2025-06-08

Align with msprog R package.

### Changed
- Function MSprog()

### Added
- Functions separate_ri_ra(), value_milestone(), is_event(), load_toy_data()

### Removed
- Function col_to_date()

---

# 0.1.1 — 2023-07-21

### Changed
- Function MSprog()

---

# 0.1.0 — 2023-03-13

### Added
- Functions MSprog(), col_to_date(), age_column(), compute_delta()

