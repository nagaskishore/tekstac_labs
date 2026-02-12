--USERS
INSERT INTO users (name, email, profile, travel_preferences, travel_constraints) VALUES
('Alice Johnson', 'alice@example.com', 'Luxury Traveler', 'luxury hotels, gourmet dining, art museums, private tours', 'no budget limit, prefers direct flights'),
('Bob Smith', 'bob@example.com', 'Budget Backpacker', 'hostels, street food, adventure sports, local transport', 'low budget, avoids expensive activities'),
('Carla Mendes', 'carla@example.com', 'Family Mom', 'family-friendly attractions, safety, kids activities, comfortable accommodation', 'child safety, no late-night events'),
('David Kim', 'david@example.com', 'Business Professional', 'business hotels, loyalty programs, fast WiFi, efficient transport', 'tight schedule, prefers business amenities'),
('Emma Brown', 'emma@example.com', 'Student', 'authentic experiences, budget stays, cultural immersion, group activities', 'limited budget, prefers youth hostels');

-- TRIPS
INSERT INTO trips (user_id, phase, title, origin, destination, trip_startdate, trip_enddate, accommodation_type, no_of_adults, no_of_children, budget, currency, trip_status, purpose, travel_preferences, travel_constraints)
VALUES
	(1, 'phase2_crewai', 'Alice Paris Luxury', 'New York', 'Paris', '2026-03-10', '2026-03-20', 'luxury', 2, 0, 5000, 'USD', 'draft', 'vacation', 'art, fine dining', 'prefer direct flights'),
	(2, 'phase2_crewai', 'Bob Thailand Adventure', 'Los Angeles', 'Bangkok', '2026-04-05', '2026-04-15', 'budget', 1, 0, 1500, 'USD', 'draft', 'adventure', 'local experiences, street food', 'low budget'),
	(3, 'phase2_crewai', 'Carla Rome Family Trip', 'Chicago', 'Rome', '2026-05-01', '2026-05-10', 'family-friendly', 2, 2, 4000, 'USD', 'draft', 'family vacation', 'safety, kids activities', 'child safety'),
	(4, 'phase2_crewai', 'David Tokyo Business Travel', 'San Francisco', 'Tokyo', '2026-06-01', '2026-06-05', 'business', 1, 0, 3000, 'USD', 'draft', 'business', 'fast WiFi, business hotels', 'tight schedule'),
	(5, 'phase2_crewai', 'Emma Barcelona Student Trip', 'Miami', 'Barcelona', '2026-07-15', '2026-07-25', 'youth hostel', 1, 0, 1000, 'USD', 'draft', 'cultural immersion', 'authentic experiences, budget stays', 'limited budget');

-- CHAT HISTORY
INSERT INTO chat_history (trip_id, user_id, role, phase, content, sequence_number)
VALUES
	(1, 1, 'user', 'phase2_crewai', 'I want a luxury trip to Paris.', 1),
	(2, 2, 'user', 'phase2_crewai', 'Looking for budget adventure in Thailand.', 1),
	(3, 3, 'user', 'phase2_crewai', 'Family trip to Rome for 5.', 1),
	(4, 4, 'user', 'phase2_crewai', 'Business travel to Tokyo.', 1),
	(5, 5, 'user', 'phase2_crewai', 'Student trip to Barcelona.', 1);

